import asyncio
import functools
import importlib
import os
import threading
import warnings
from typing import Any, Callable, Optional, Union
from urllib.parse import urlparse, urlunparse

from pydantic import Field, model_validator
from redis.asyncio import Redis
from redis.credentials import CredentialProvider
from typing_extensions import Self, TypeAlias

from prefect.settings.base import (
    PrefectBaseSettings,
    build_settings_config,  # type: ignore[reportPrivateUsage]
)
from prefect_redis.connection import aclose_redis_client, redis_from_url

_UNSET: Any = object()


class RedisMessagingSettings(PrefectBaseSettings):
    """Settings for connecting to Redis.

    Connection can be configured either via a single `url` field
    (e.g. `redis://user:pass@host:6379/0`) or with the individual
    `host`/`port`/`db`/… fields.  When `url` is set it takes
    precedence and the discrete fields are ignored.

    Environment variable: `PREFECT_REDIS_MESSAGING_URL`
    """

    model_config = build_settings_config(
        (
            "redis",
            "messaging",
        ),
        frozen=True,
    )

    url: Optional[str] = Field(
        default=None,
        description=(
            "Full Redis URL (e.g. redis://user:pass@host:6379/0 or "
            "rediss://… for TLS). Also supports redis+sentinel:// and "
            "rediss+sentinel:// for Redis Sentinel; the Sentinel schemes "
            "accept a comma-separated list of members and a master group "
            "name, e.g. "
            "redis+sentinel://sentinel-a:26379,sentinel-b:26379/mymaster. "
            "When set, host/port/db/username/password/ssl are ignored."
        ),
    )
    host: str = Field(default="localhost")
    port: int = Field(default=6379)
    db: int = Field(default=0)
    username: str = Field(default="default")
    password: str = Field(default="")
    health_check_interval: int = Field(
        default=20,
        description="Health check interval for pinging the server; defaults to 20 seconds.",
    )
    ssl: bool = Field(
        default=False,
        description="Whether to use SSL for the Redis connection",
    )
    socket_timeout: Optional[float] = Field(
        default=None,
        description=(
            "Timeout in seconds for socket read operations. "
            "None means no timeout (preserves pre-redis-py-8 behavior)."
        ),
    )
    socket_connect_timeout: Optional[float] = Field(
        default=None,
        description=(
            "Timeout in seconds for socket connect operations. "
            "None means no timeout (preserves pre-redis-py-8 behavior)."
        ),
    )
    protocol: int = Field(
        default=2,
        description=(
            "RESP protocol version. Defaults to 2 for compatibility "
            "with older Redis servers and proxies."
        ),
    )
    credential_provider_factory: Optional[str] = Field(
        default=None,
        description=(
            "Import path to a zero-argument callable returning a "
            "`redis.credentials.CredentialProvider` instance (e.g. "
            "`package.module:create_provider`). When set, the provider takes "
            "precedence over static username/password and URL credentials."
        ),
    )

    _DISCRETE_FIELDS: frozenset[str] = frozenset(
        {"host", "port", "db", "username", "password", "ssl"}
    )
    _STATIC_CREDENTIAL_FIELDS: frozenset[str] = frozenset({"username", "password"})

    @model_validator(mode="after")
    def _warn_url_overrides_discrete_fields(self) -> Self:
        conflicting = self._DISCRETE_FIELDS & self.model_fields_set
        if self.url is not None and conflicting:
            warnings.warn(
                f"Redis URL is set; the following fields are ignored: "
                f"{', '.join(sorted(conflicting))}",
            )
        return self

    @model_validator(mode="after")
    def _warn_credential_provider_overrides_static_credentials(self) -> Self:
        if self.credential_provider_factory is not None and self.url is None:
            conflicting = self._STATIC_CREDENTIAL_FIELDS & self.model_fields_set
            if conflicting:
                warnings.warn(
                    "Redis credential_provider_factory is set; the following "
                    "static credential fields are ignored: "
                    f"{', '.join(sorted(conflicting))}",
                )
        return self


CacheKey: TypeAlias = tuple[
    Callable[..., Any],
    tuple[Any, ...],
    tuple[tuple[str, Any], ...],
    Union[asyncio.AbstractEventLoop, None],
]

_client_cache: dict[CacheKey, Redis] = {}
_credential_provider_cache: dict[tuple[str, int], CredentialProvider] = {}
_credential_provider_lock = threading.Lock()


def _load_credential_provider_factory(
    factory_path: str,
) -> Callable[[], CredentialProvider]:
    """Import and validate a zero-argument `CredentialProvider` factory callable."""
    stripped = factory_path.strip()
    if not stripped:
        raise ValueError(
            "Invalid credential_provider_factory: path must be a non-empty string."
        )

    if ":" in stripped:
        module_name, _, attr_path = stripped.partition(":")
        if not module_name or not attr_path or ":" in attr_path:
            raise ValueError(
                f"Invalid credential_provider_factory path {factory_path!r}. "
                "Expected 'package.module:callable'."
            )
    elif "." in stripped:
        module_name, attr_path = stripped.rsplit(".", 1)
        if not module_name or not attr_path:
            raise ValueError(
                f"Invalid credential_provider_factory path {factory_path!r}. "
                "Expected 'package.module:callable'."
            )
    else:
        raise ValueError(
            f"Invalid credential_provider_factory path {factory_path!r}. "
            "Expected 'package.module:callable'."
        )

    try:
        obj: Any = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"Failed to import credential_provider_factory module {module_name!r} "
            f"from {factory_path!r}: {exc}"
        ) from exc

    for attr in attr_path.split("."):
        if not attr:
            raise ValueError(
                f"Invalid credential_provider_factory path {factory_path!r}. "
                "Expected 'package.module:callable'."
            )
        try:
            obj = getattr(obj, attr)
        except AttributeError as exc:
            raise AttributeError(
                f"Credential provider factory {factory_path!r} could not be "
                f"resolved: {exc}"
            ) from exc

    if not callable(obj):
        raise TypeError(
            f"Credential provider factory {factory_path!r} must be callable, "
            f"got {type(obj).__name__}."
        )

    return obj


def _resolve_credential_provider(factory_path: str) -> CredentialProvider:
    """Resolve and cache a `CredentialProvider` once per process (PID-scoped)."""
    normalized_path = factory_path.strip()
    if not normalized_path:
        raise ValueError(
            "Invalid credential_provider_factory: path must be a non-empty string."
        )

    cache_key = (normalized_path, os.getpid())
    with _credential_provider_lock:
        if cache_key in _credential_provider_cache:
            return _credential_provider_cache[cache_key]

        factory = _load_credential_provider_factory(normalized_path)
        provider = factory()
        if not isinstance(provider, CredentialProvider):
            raise TypeError(
                f"Credential provider factory {factory_path!r} must return a "
                f"redis.credentials.CredentialProvider instance, "
                f"got {type(provider).__name__}."
            )
        _credential_provider_cache[cache_key] = provider
        return provider


def clear_cached_credential_providers() -> None:
    """Clear cached `CredentialProvider` instances."""
    with _credential_provider_lock:
        _credential_provider_cache.clear()


def is_cluster_url(url: str) -> bool:
    """Return True if the URL uses the Redis Cluster scheme."""
    return url.partition("://")[0].lower() in {"redis+cluster", "rediss+cluster"}


def normalize_cluster_url(url: str) -> str:
    """Return a redis-py compatible URL for Redis Cluster connections."""
    parsed = urlparse(url)
    if parsed.scheme not in {"redis+cluster", "rediss+cluster"}:
        return url

    return urlunparse(parsed._replace(scheme=parsed.scheme.replace("+cluster", "")))


@functools.cache
def _get_redis_messaging_url() -> str | None:
    return RedisMessagingSettings().url


def cluster_key_prefix(prefix: str, url: str | None = None) -> str:
    """Return a key prefix, hash-tagged when configured for Redis Cluster."""
    url = url or _get_redis_messaging_url()
    if url and is_cluster_url(url):
        return f"{{{prefix}}}"
    return prefix


def redis_key(prefix: str, suffix: str, url: str | None = None) -> str:
    """Return a Redis key rooted at a cluster-aware prefix."""
    return f"{cluster_key_prefix(prefix, url=url)}:{suffix}"


def _raise_cluster_not_supported() -> None:
    raise NotImplementedError(
        "Redis Cluster URLs are detected but not enabled yet. "
        "Cluster support requires hash-slot-safe keys across the Redis-backed "
        "messaging, ordering, lease storage, and cleanup queue subsystems."
    )


def _running_loop() -> Union[asyncio.AbstractEventLoop, None]:
    try:
        return asyncio.get_running_loop()
    except RuntimeError as e:
        if "no running event loop" in str(e):
            return None
        raise


def cached(fn: Callable[..., Any]) -> Callable[..., Any]:
    @functools.wraps(fn)
    def cached_fn(*args: Any, **kwargs: Any) -> Redis:
        key = (fn, args, tuple(kwargs.items()), _running_loop())
        if key not in _client_cache:
            _client_cache[key] = fn(*args, **kwargs)
        return _client_cache[key]

    return cached_fn


def close_all_cached_connections() -> None:
    """Close all cached Redis connections.

    Sentinel-backed clients hold one extra Redis client per Sentinel daemon, so
    the daemon-aware `aclose_redis_client` is used rather than a bare
    `client.aclose()`.
    """
    loop: Union[asyncio.AbstractEventLoop, None]

    for (_, _, _, loop), client in _client_cache.items():
        if not loop or (loop and loop.is_closed()):
            continue
        loop.run_until_complete(client.connection_pool.disconnect())
        loop.run_until_complete(aclose_redis_client(client))


async def clear_cached_clients() -> None:
    """Clear cached Redis clients and the messaging URL lookup."""
    _get_redis_messaging_url.cache_clear()
    _client_cache.clear()


def _build_async_redis_client(
    *,
    url: Union[str, None],
    host: str,
    port: int,
    db: int,
    username: str,
    password: str,
    health_check_interval: int,
    ssl: bool,
    credential_provider: Union[CredentialProvider, None],
    explicit_static_credentials: bool,
    options: dict[str, Any],
) -> Redis:
    """Build an async `Redis` client from URL or discrete connection parameters."""
    if credential_provider is not None:
        if explicit_static_credentials:
            warnings.warn(
                "Redis credential_provider is set; explicit username and "
                "password credentials are ignored."
            )
        options.pop("username", None)
        options.pop("password", None)
        options["credential_provider"] = credential_provider

    if url:
        if is_cluster_url(url):
            _raise_cluster_not_supported()
        return redis_from_url(
            url,
            asynchronous=True,
            health_check_interval=health_check_interval,
            **options,
        )

    if credential_provider is not None:
        return Redis(
            host=host,
            port=port,
            db=db,
            health_check_interval=health_check_interval,
            ssl=ssl,
            **options,
        )

    return Redis(
        host=host,
        port=port,
        db=db,
        password=password,
        username=username,
        health_check_interval=health_check_interval,
        ssl=ssl,
        **options,
    )


@cached
def get_async_redis_client(
    url: Union[str, None] = None,
    host: Union[str, None] = None,
    port: Union[int, None] = None,
    db: Union[int, None] = None,
    password: Union[str, None] = None,
    username: Union[str, None] = None,
    health_check_interval: Union[int, None] = None,
    decode_responses: bool = True,
    ssl: Union[bool, None] = None,
    socket_timeout: Union[float, None, Any] = _UNSET,
    socket_connect_timeout: Union[float, None, Any] = _UNSET,
    protocol: Union[int, None] = None,
    credential_provider: Union[CredentialProvider, None] = None,
) -> Redis:
    """Retrieves an async Redis client.

    When a `url` is provided (or configured via `PREFECT_REDIS_MESSAGING_URL`)
    the discrete host/port/… arguments are ignored. Standalone URLs are passed
    to `Redis.from_url`; `redis+sentinel://` and `rediss+sentinel://` URLs
    resolve the current master through the listed Sentinel daemons and follow
    failover automatically. URL query options override the settings-derived
    connection defaults. Redis Cluster URLs are detected but intentionally not
    enabled yet.

    When `credential_provider` is provided (or configured via
    `PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY`), it takes precedence
    over static `username`/`password` and embedded URL credentials.

    Args:
        url: Full Redis URL (e.g. `redis://localhost:6379/0` or
            `redis+sentinel://sentinel-a:26379,sentinel-b:26379/mymaster`).
        host: The host location.
        port: The port to connect to the host with.
        db: The Redis database to interact with.
        password: The password for the redis host
        username: Username for the redis instance
        health_check_interval: Health check interval in seconds.
        decode_responses: Whether to decode binary responses from Redis to
            unicode strings.
        ssl: Whether to use SSL for the connection.
        socket_timeout: Timeout for socket read operations (None = no timeout).
        socket_connect_timeout: Timeout for socket connect (None = no timeout).
        protocol: RESP protocol version (default 2 from settings).
        credential_provider: Optional `redis.credentials.CredentialProvider`
            instance overriding `PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY`.

    Returns:
        Redis: a Redis client
    """
    settings = RedisMessagingSettings()

    resolved_credential_provider = credential_provider
    if (
        resolved_credential_provider is None
        and settings.credential_provider_factory is not None
    ):
        resolved_credential_provider = _resolve_credential_provider(
            settings.credential_provider_factory
        )

    resolved_socket_timeout = (
        settings.socket_timeout if socket_timeout is _UNSET else socket_timeout
    )
    resolved_socket_connect_timeout = (
        settings.socket_connect_timeout
        if socket_connect_timeout is _UNSET
        else socket_connect_timeout
    )
    resolved_protocol = protocol if protocol is not None else settings.protocol

    return _build_async_redis_client(
        url=url or settings.url,
        host=host or settings.host,
        port=port if port is not None else settings.port,
        db=db if db is not None else settings.db,
        username=username or settings.username,
        password=password or settings.password,
        health_check_interval=(
            health_check_interval
            if health_check_interval is not None
            else settings.health_check_interval
        ),
        ssl=ssl if ssl is not None else settings.ssl,
        credential_provider=resolved_credential_provider,
        explicit_static_credentials=(username is not None or password is not None),
        options={
            "decode_responses": decode_responses,
            "socket_timeout": resolved_socket_timeout,
            "socket_connect_timeout": resolved_socket_connect_timeout,
            "protocol": resolved_protocol,
        },
    )


@cached
def async_redis_from_settings(
    settings: RedisMessagingSettings, **options: Any
) -> Redis:
    """Create a cached async Redis client from a `RedisMessagingSettings` instance."""
    resolved_options: dict[str, Any] = {
        "decode_responses": True,
        "socket_timeout": settings.socket_timeout,
        "socket_connect_timeout": settings.socket_connect_timeout,
        "protocol": settings.protocol,
        **options,
    }

    explicit_static_credentials = (
        resolved_options.get("username") is not None
        or resolved_options.get("password") is not None
    )

    resolved_credential_provider = resolved_options.pop("credential_provider", None)
    if (
        resolved_credential_provider is None
        and settings.credential_provider_factory is not None
    ):
        resolved_credential_provider = _resolve_credential_provider(
            settings.credential_provider_factory
        )

    return _build_async_redis_client(
        url=settings.url,
        host=settings.host,
        port=settings.port,
        db=settings.db,
        username=settings.username,
        password=settings.password,
        health_check_interval=settings.health_check_interval,
        ssl=settings.ssl,
        credential_provider=resolved_credential_provider,
        explicit_static_credentials=explicit_static_credentials,
        options=resolved_options,
    )
