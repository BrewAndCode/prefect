import asyncio
import os
import sys
import uuid
import warnings
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import redis.asyncio
from prefect_redis.client import (
    RedisMessagingSettings,
    _client_cache,
    async_redis_from_settings,
    clear_cached_clients,
    clear_cached_credential_providers,
    close_all_cached_connections,
    cluster_key_prefix,
    get_async_redis_client,
    is_cluster_url,
    normalize_cluster_url,
    redis_key,
)
from redis.asyncio import Redis
from redis.cluster import key_slot
from redis.credentials import CredentialProvider, UsernamePasswordCredentialProvider


def test_redis_settings_defaults(isolated_redis_db_number: int):
    """Test that RedisSettings has expected defaults"""
    settings = RedisMessagingSettings()
    assert settings.url is None
    assert settings.host == "localhost"
    assert settings.port == 6379
    # Note: we override the db number in the conftest
    # to isolate redis db for xdist workers
    assert settings.db == isolated_redis_db_number
    assert settings.username == "default"
    assert settings.password == ""
    assert settings.health_check_interval == 20
    assert settings.ssl is False


def test_redis_settings_url():
    """Test that url field can be set directly"""
    settings = RedisMessagingSettings(url="redis://myhost:6380/2")
    assert settings.url == "redis://myhost:6380/2"


def test_redis_settings_url_from_env(monkeypatch: pytest.MonkeyPatch):
    """Test that url can be configured via environment variable"""
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_URL", "redis://envhost:6381/3")
    settings = RedisMessagingSettings()
    assert settings.url == "redis://envhost:6381/3"


def test_cluster_url_detection():
    assert is_cluster_url("redis+cluster://redis.example.com:6379")
    assert is_cluster_url("rediss+cluster://redis.example.com:6379")
    assert not is_cluster_url("redis://redis.example.com:6379")
    assert not is_cluster_url("rediss://redis.example.com:6379")
    assert not is_cluster_url("redis://host,[::1]:6379")


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (
            "redis+cluster://redis.example.com:6379",
            "redis://redis.example.com:6379",
        ),
        (
            "rediss+cluster://user:pass@redis.example.com:6380/0?protocol=3",
            "rediss://user:pass@redis.example.com:6380/0?protocol=3",
        ),
        (
            "redis://redis.example.com:6379/0",
            "redis://redis.example.com:6379/0",
        ),
    ],
)
def test_normalize_cluster_url(url: str, expected: str):
    assert normalize_cluster_url(url) == expected


def test_cluster_key_prefix_hash_tags_cluster_urls():
    assert (
        cluster_key_prefix(
            "prefect:events", url="redis+cluster://redis.example.com:6379"
        )
        == "{prefect:events}"
    )
    assert (
        cluster_key_prefix(
            "prefect:events", url="rediss+cluster://redis.example.com:6379"
        )
        == "{prefect:events}"
    )
    assert (
        cluster_key_prefix("prefect:events", url="redis://redis.example.com:6379")
        == "prefect:events"
    )


@patch("prefect_redis.client.RedisMessagingSettings")
async def test_cluster_key_prefix_caches_settings_until_clients_are_cleared(
    settings: MagicMock,
):
    settings.return_value.url = None

    assert cluster_key_prefix("prefect:events") == "prefect:events"
    assert cluster_key_prefix("prefect:events") == "prefect:events"
    settings.assert_called_once_with()

    await clear_cached_clients()

    assert cluster_key_prefix("prefect:events") == "prefect:events"
    assert settings.call_count == 2


def test_redis_key_uses_cluster_aware_prefix():
    assert (
        redis_key(
            "prefect:events",
            "stream",
            url="redis+cluster://redis.example.com:6379",
        )
        == "{prefect:events}:stream"
    )
    assert (
        redis_key("prefect:events", "stream", url="redis://redis.example.com:6379")
        == "prefect:events:stream"
    )


def test_cluster_keys_share_hash_slot():
    keys = [
        redis_key(
            "prefect:events",
            suffix,
            url="redis+cluster://redis.example.com:6379",
        )
        for suffix in ["stream", "dlq", "dedupe:abc"]
    ]
    assert len({key_slot(key.encode()) for key in keys}) == 1


def test_redis_settings_url_warns_on_conflicting_fields():
    """When url and discrete fields are both set, warn that discrete fields are ignored"""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RedisMessagingSettings(url="redis://myhost:6380/2", host="other-host")

    assert len(caught) == 1
    assert "host" in str(caught[0].message)
    assert "ignored" in str(caught[0].message)


def test_redis_settings_url_no_warning_with_defaults(monkeypatch: pytest.MonkeyPatch):
    """No warning when url is set but discrete fields are all defaults"""
    monkeypatch.delenv("PREFECT_REDIS_MESSAGING_DB", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RedisMessagingSettings(url="redis://myhost:6380/2")

    assert len(caught) == 0


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({"ssl": False}, {"ssl": False}),
        ({"port": 0}, {"port": 0}),
        ({"db": 0}, {"db": 0}),
        ({"health_check_interval": 0}, {"health_check_interval": 0}),
    ],
)
def test_get_async_redis_client_preserves_explicit_falsy_overrides(
    monkeypatch: pytest.MonkeyPatch,
    overrides: dict[str, bool | int],
    expected: dict[str, bool | int],
):
    settings = RedisMessagingSettings(
        ssl=True, port=6380, db=1, health_check_interval=30
    )
    redis = MagicMock()
    monkeypatch.setattr("prefect_redis.client.RedisMessagingSettings", lambda: settings)
    monkeypatch.setattr("prefect_redis.client.Redis", redis)
    _client_cache.clear()

    try:
        get_async_redis_client(**overrides)

        for key, value in expected.items():
            assert redis.call_args.kwargs[key] == value
    finally:
        _client_cache.clear()


def test_get_async_redis_client_none_overrides_inherit_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    settings = RedisMessagingSettings(
        ssl=True, port=6380, db=1, health_check_interval=30
    )
    redis = MagicMock()
    monkeypatch.setattr("prefect_redis.client.RedisMessagingSettings", lambda: settings)
    monkeypatch.setattr("prefect_redis.client.Redis", redis)
    _client_cache.clear()

    try:
        get_async_redis_client()

        expected = {
            "ssl": True,
            "port": 6380,
            "db": 1,
            "health_check_interval": 30,
        }
        for key, value in expected.items():
            assert redis.call_args.kwargs[key] == value
    finally:
        _client_cache.clear()


def test_get_async_redis_client_url_preserves_explicit_zero_health_check_interval(
    monkeypatch: pytest.MonkeyPatch,
):
    redis_from_url = MagicMock()
    monkeypatch.setattr("prefect_redis.client.redis_from_url", redis_from_url)
    _client_cache.clear()

    try:
        get_async_redis_client(url="redis://localhost:6379/0", health_check_interval=0)

        assert redis_from_url.call_args.kwargs["health_check_interval"] == 0
    finally:
        _client_cache.clear()


async def test_get_async_redis_client_defaults():
    """Test that get_async_redis_client creates client with default settings"""
    client = get_async_redis_client()
    assert isinstance(client, Redis)
    assert client.connection_pool.connection_kwargs["host"] == "localhost"
    assert client.connection_pool.connection_kwargs["port"] == 6379
    await client.aclose()


async def test_get_async_redis_client_custom_params():
    """Test that get_async_redis_client respects custom parameters"""
    client = get_async_redis_client(
        host="custom.host",
        port=6380,
        db=1,
        username="custom_user",
        password="secret",
    )
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["host"] == "custom.host"
    assert conn_kwargs["port"] == 6380
    assert conn_kwargs["db"] == 1
    assert conn_kwargs["username"] == "custom_user"
    assert conn_kwargs["password"] == "secret"
    await client.aclose()


async def test_get_async_redis_client_with_url():
    """Test that get_async_redis_client uses from_url when url is provided"""
    client = get_async_redis_client(url="redis://urlhost:6382/4")
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["host"] == "urlhost"
    assert conn_kwargs["port"] == 6382
    assert conn_kwargs["db"] == 4
    await client.aclose()


async def test_get_async_redis_client_with_cluster_url_raises():
    """Cluster URLs are detected but not enabled until key work is complete."""
    _client_cache.clear()
    with pytest.raises(NotImplementedError, match="Redis Cluster URLs"):
        get_async_redis_client(url="redis+cluster://clusterhost:7000")


async def test_get_async_redis_client_url_with_credentials():
    """Test that Redis URL with embedded credentials works"""
    client = get_async_redis_client(url="redis://myuser:mypass@urlhost:6382/4")
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["host"] == "urlhost"
    assert conn_kwargs["port"] == 6382
    assert conn_kwargs["db"] == 4
    assert conn_kwargs["username"] == "myuser"
    assert conn_kwargs["password"] == "mypass"
    await client.aclose()


async def test_get_async_redis_client_url_from_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    """Test that url from settings is used when no explicit url param is passed"""
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_URL", "redis://settingshost:6383/5")
    # Clear cache so the env var is picked up by a fresh RedisMessagingSettings
    _client_cache.clear()
    try:
        client = get_async_redis_client()
        conn_kwargs = client.connection_pool.connection_kwargs
        assert conn_kwargs["host"] == "settingshost"
        assert conn_kwargs["port"] == 6383
        assert conn_kwargs["db"] == 5
        await client.aclose()
    finally:
        # The no-args cache entry now points at the URL-based client.
        # Clear it so subsequent tests (and the autouse redis fixture)
        # get a client matching the real env again.
        _client_cache.clear()


async def test_get_async_redis_client_explicit_url_overrides_settings(
    monkeypatch: pytest.MonkeyPatch,
):
    """Explicit url param takes precedence over PREFECT_REDIS_MESSAGING_URL"""
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_URL", "redis://settingshost:6383/5")
    client = get_async_redis_client(url="redis://explicithost:6384/6")
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["host"] == "explicithost"
    assert conn_kwargs["port"] == 6384
    assert conn_kwargs["db"] == 6
    await client.aclose()


async def test_async_redis_from_settings():
    """Test creating Redis client from settings object"""
    settings = RedisMessagingSettings(
        host="settings.host",
        port=6381,
        username="settings_user",
    )
    client = async_redis_from_settings(settings)
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["host"] == "settings.host"
    assert conn_kwargs["port"] == 6381
    assert conn_kwargs["username"] == "settings_user"
    await client.aclose()


async def test_async_redis_from_settings_with_url():
    """Test creating Redis client from settings with url"""
    settings = RedisMessagingSettings(url="redis://fromurl:6385/7")
    client = async_redis_from_settings(settings)
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["host"] == "fromurl"
    assert conn_kwargs["port"] == 6385
    assert conn_kwargs["db"] == 7
    await client.aclose()


async def test_async_redis_from_settings_with_cluster_url_raises():
    """Settings URL cluster support is detection-only for now."""
    _client_cache.clear()
    settings = RedisMessagingSettings(url="rediss+cluster://fromurl:6385")
    with pytest.raises(NotImplementedError, match="Redis Cluster URLs"):
        async_redis_from_settings(settings)


def test_redis_settings_connection_defaults():
    """Settings default to socket_timeout=None, socket_connect_timeout=None, protocol=2."""
    settings = RedisMessagingSettings()
    assert settings.socket_timeout is None
    assert settings.socket_connect_timeout is None
    assert settings.protocol == 2


def test_redis_settings_connection_from_env(monkeypatch: pytest.MonkeyPatch):
    """Connection settings can be configured via environment variables."""
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_SOCKET_TIMEOUT", "10.0")
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_SOCKET_CONNECT_TIMEOUT", "3.5")
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_PROTOCOL", "3")
    settings = RedisMessagingSettings()
    assert settings.socket_timeout == 10.0
    assert settings.socket_connect_timeout == 3.5
    assert settings.protocol == 3


async def test_get_async_redis_client_default_socket_timeout():
    """Default clients have socket_timeout=None (no timeout) for redis-py 8 compat."""
    _client_cache.clear()
    client = get_async_redis_client()
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs.get("socket_timeout") is None
    assert conn_kwargs.get("socket_connect_timeout") is None
    await client.aclose()
    _client_cache.clear()


async def test_get_async_redis_client_default_protocol():
    """Default clients use protocol=2 for older Redis/proxy compatibility."""
    _client_cache.clear()
    client = get_async_redis_client()
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs.get("protocol") == 2
    await client.aclose()
    _client_cache.clear()


async def test_get_async_redis_client_explicit_socket_timeout():
    """Explicit socket_timeout overrides the settings default."""
    _client_cache.clear()
    client = get_async_redis_client(socket_timeout=30.0, socket_connect_timeout=5.0)
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["socket_timeout"] == 30.0
    assert conn_kwargs["socket_connect_timeout"] == 5.0
    await client.aclose()
    _client_cache.clear()


async def test_get_async_redis_client_url_passes_socket_timeout():
    """socket_timeout/protocol are passed through the from_url path."""
    _client_cache.clear()
    client = get_async_redis_client(url="redis://localhost:6379/0")
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs.get("socket_timeout") is None
    assert conn_kwargs.get("socket_connect_timeout") is None
    assert conn_kwargs.get("protocol") == 2
    await client.aclose()
    _client_cache.clear()


async def test_get_async_redis_client_url_query_overrides_keyword_defaults():
    """URL query params override keyword defaults (redis-py from_url behavior)."""
    _client_cache.clear()
    client = get_async_redis_client(
        url="redis://localhost:6379/0?socket_timeout=7&socket_connect_timeout=3"
    )
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["socket_timeout"] == 7
    assert conn_kwargs["socket_connect_timeout"] == 3
    await client.aclose()
    _client_cache.clear()


async def test_async_redis_from_settings_passes_connection_defaults():
    """async_redis_from_settings passes socket_timeout/protocol from settings."""
    _client_cache.clear()
    settings = RedisMessagingSettings()
    client = async_redis_from_settings(settings)
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs.get("socket_timeout") is None
    assert conn_kwargs.get("socket_connect_timeout") is None
    assert conn_kwargs.get("protocol") == 2
    await client.aclose()
    _client_cache.clear()


async def test_async_redis_from_settings_options_override():
    """Options kwargs override settings defaults in async_redis_from_settings."""
    _client_cache.clear()
    settings = RedisMessagingSettings()
    client = async_redis_from_settings(settings, socket_timeout=15.0, protocol=3)
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs["socket_timeout"] == 15.0
    assert conn_kwargs["protocol"] == 3
    await client.aclose()
    _client_cache.clear()


async def test_async_redis_from_settings_url_with_connection_defaults():
    """async_redis_from_settings with url passes connection defaults."""
    _client_cache.clear()
    settings = RedisMessagingSettings(url="redis://localhost:6379/0")
    client = async_redis_from_settings(settings)
    conn_kwargs = client.connection_pool.connection_kwargs
    assert conn_kwargs.get("socket_timeout") is None
    assert conn_kwargs.get("protocol") == 2
    await client.aclose()
    _client_cache.clear()


@patch("prefect_redis.client._client_cache")
def test_close_all_cached_connections(mock_cache):
    """Test that close_all_cached_connections properly closes all clients"""
    mock_client = MagicMock()
    mock_loop = MagicMock()
    mock_loop.is_closed.return_value = False

    # Mock the coroutines that would be awaited
    mock_loop.run_until_complete.return_value = None

    mock_cache.items.return_value = [((None, None, None, mock_loop), mock_client)]

    close_all_cached_connections()

    # Verify run_until_complete was called twice (for disconnect and close)
    assert mock_loop.run_until_complete.call_count == 2


def test_close_all_cached_connections_closes_sentinel_daemons():
    """A cached Sentinel-backed client holds one Redis client per Sentinel
    daemon; cache teardown must close those too, not just the master client."""
    _client_cache.clear()
    loop = asyncio.new_event_loop()
    try:

        async def build() -> Redis:
            return get_async_redis_client(
                url="redis+sentinel://s1:26379,s2:26379/mymaster"
            )

        client = loop.run_until_complete(build())
        daemons = list(client.connection_pool.sentinel_manager.sentinels)
        assert daemons
        with patch.object(redis.asyncio.Redis, "aclose", autospec=True) as mock_aclose:
            close_all_cached_connections()
        closed = {id(call.args[0]) for call in mock_aclose.call_args_list}
        assert id(client) in closed
        for daemon in daemons:
            assert id(daemon) in closed
    finally:
        _client_cache.clear()
        loop.close()


class _RecordingCredentialProvider(CredentialProvider):
    def __init__(self, username: str, password: str) -> None:
        self.username = username
        self.password = password
        self.calls = 0

    def get_credentials(self) -> tuple[str, str]:
        self.calls += 1
        return self.username, self.password

    async def get_credentials_async(self) -> tuple[str, str]:
        self.calls += 1
        return self.username, self.password


class _FactoryHolder:
    @classmethod
    def create_classmethod(cls) -> CredentialProvider:
        return UsernamePasswordCredentialProvider("holder_user", "holder_pass")


_LIVE_PROVIDERS: list[CredentialProvider] = []
_LIVE_FACTORY_CALLS = 0


def _create_live_provider() -> CredentialProvider:
    global _LIVE_FACTORY_CALLS
    _LIVE_FACTORY_CALLS += 1
    return _LIVE_PROVIDERS[0]


def _create_provider_for_dotted_import() -> CredentialProvider:
    return UsernamePasswordCredentialProvider("dotted_user", "dotted_pass")


_NON_CALLABLE_TARGET = "not-callable"


def _raising_factory() -> CredentialProvider:
    raise RuntimeError("factory boom")


def _invalid_return_factory() -> object:
    return "not-a-credential-provider"


def test_credential_provider_factory_setting_default_and_env(
    monkeypatch: pytest.MonkeyPatch,
):
    assert RedisMessagingSettings().credential_provider_factory is None

    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        "pkg.module:create_provider",
    )
    settings = RedisMessagingSettings()
    assert settings.credential_provider_factory == "pkg.module:create_provider"


def test_credential_provider_factory_with_url_emits_no_warning(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.delenv("PREFECT_REDIS_MESSAGING_DB", raising=False)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        settings = RedisMessagingSettings(
            url="redis://myhost:6380/2",
            credential_provider_factory="pkg.module:create_provider",
        )
    assert settings.url == "redis://myhost:6380/2"
    assert settings.credential_provider_factory == "pkg.module:create_provider"
    assert len(caught) == 0


def test_credential_provider_factory_warns_when_static_credentials_set(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_USERNAME", "static_user")
    monkeypatch.setenv("PREFECT_REDIS_MESSAGING_PASSWORD", "static_pass")
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        RedisMessagingSettings(
            credential_provider_factory="pkg.module:create_provider",
        )
    assert len(caught) == 1
    message = str(caught[0].message)
    assert "credential_provider_factory" in message
    assert "username" in message
    assert "password" in message


def test_credential_provider_factory_is_lazily_imported(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        "nonexistent_module_12345:create_provider",
    )
    # Constructing settings must not import the factory.
    settings = RedisMessagingSettings()
    assert (
        settings.credential_provider_factory
        == "nonexistent_module_12345:create_provider"
    )


@pytest.mark.parametrize(
    ("factory_path", "exc_type", "match"),
    [
        ("", ValueError, "non-empty"),
        ("   ", ValueError, "non-empty"),
        ("no_separator", ValueError, "Invalid credential_provider_factory"),
        (":missing_module", ValueError, "Invalid credential_provider_factory"),
        ("missing_attr:", ValueError, "Invalid credential_provider_factory"),
        ("mod:attr:extra", ValueError, "Invalid credential_provider_factory"),
        ("nonexistent_pkg_xyz.mod:create", ImportError, "Failed to import"),
        (
            "prefect_redis.client:missing_factory_attr",
            AttributeError,
            "could not be resolved",
        ),
        (f"{__name__}:_NON_CALLABLE_TARGET", TypeError, "must be callable"),
        (f"{__name__}:_raising_factory", RuntimeError, "factory boom"),
        (
            f"{__name__}:_invalid_return_factory",
            TypeError,
            "CredentialProvider instance",
        ),
    ],
)
def test_invalid_credential_provider_factory_fails_clearly(
    monkeypatch: pytest.MonkeyPatch,
    factory_path: str,
    exc_type: type[Exception],
    match: str,
):
    _client_cache.clear()
    clear_cached_credential_providers()
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY", factory_path
    )
    try:
        with pytest.raises(exc_type, match=match):
            get_async_redis_client()
    finally:
        _client_cache.clear()
        clear_cached_credential_providers()


@pytest.mark.parametrize(
    "factory_path",
    [
        f"{__name__}:_create_provider_for_dotted_import",
        f"{__name__}._create_provider_for_dotted_import",
        f"{__name__}:_FactoryHolder.create_classmethod",
    ],
)
async def test_credential_provider_factory_supported_import_formats(
    monkeypatch: pytest.MonkeyPatch,
    factory_path: str,
):
    _client_cache.clear()
    clear_cached_credential_providers()
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY", factory_path
    )
    try:
        client = get_async_redis_client()
        conn_kwargs = client.connection_pool.connection_kwargs
        assert isinstance(conn_kwargs["credential_provider"], CredentialProvider)
        assert conn_kwargs.get("username") is None
        assert conn_kwargs.get("password") is None
        await client.aclose()
    finally:
        _client_cache.clear()
        clear_cached_credential_providers()


async def test_credential_provider_cached_once_per_process_across_client_clears(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
):
    module_name = f"_test_provider_mod_{uuid.uuid4().hex}"
    module_file = tmp_path / f"{module_name}.py"
    module_file.write_text(
        "from redis.credentials import UsernamePasswordCredentialProvider\n"
        "CALLS = 0\n"
        "def create_provider():\n"
        "    global CALLS\n"
        "    CALLS += 1\n"
        "    return UsernamePasswordCredentialProvider('u', f'p-{CALLS}')\n"
    )
    monkeypatch.syspath_prepend(str(tmp_path))
    _client_cache.clear()
    clear_cached_credential_providers()
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        f"{module_name}:create_provider",
    )

    try:
        client1 = get_async_redis_client()
        client2 = async_redis_from_settings(RedisMessagingSettings())
        provider1 = client1.connection_pool.connection_kwargs["credential_provider"]
        provider2 = client2.connection_pool.connection_kwargs["credential_provider"]
        assert provider1 is provider2

        # Whitespace around factory path must hit the exact same cache entry.
        monkeypatch.setenv(
            "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
            f" {module_name}:create_provider ",
        )
        await clear_cached_clients()
        client_ws = get_async_redis_client()
        assert (
            client_ws.connection_pool.connection_kwargs["credential_provider"]
            is provider1
        )
        assert sys.modules[module_name].CALLS == 1

        # Clearing client cache on reconnect must preserve the process-level provider.
        monkeypatch.setenv(
            "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
            f"{module_name}:create_provider",
        )
        await clear_cached_clients()
        client3 = get_async_redis_client()
        provider3 = client3.connection_pool.connection_kwargs["credential_provider"]
        assert provider3 is provider1
        assert sys.modules[module_name].CALLS == 1

        # Forking / a new PID gets its own provider instance rather than sharing parent's.
        monkeypatch.setattr(os, "getpid", lambda: 999999)
        await clear_cached_clients()
        client4 = get_async_redis_client()
        provider4 = client4.connection_pool.connection_kwargs["credential_provider"]
        assert provider4 is not provider1
        assert sys.modules[module_name].CALLS == 2

        await client1.aclose()
        await client2.aclose()
        await client_ws.aclose()
        await client3.aclose()
        await client4.aclose()
    finally:
        _client_cache.clear()
        clear_cached_credential_providers()
        sys.modules.pop(module_name, None)


async def test_static_credentials_ignored_with_warning_when_provider_active(
    monkeypatch: pytest.MonkeyPatch,
):
    _client_cache.clear()
    clear_cached_credential_providers()
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        f"{__name__}:_create_provider_for_dotted_import",
    )

    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = get_async_redis_client(username="explicit_user", password="secret")

        assert any(
            "credential_provider" in str(w.message) and "ignored" in str(w.message)
            for w in caught
        )
        conn_kwargs = client.connection_pool.connection_kwargs
        assert isinstance(conn_kwargs["credential_provider"], CredentialProvider)
        assert conn_kwargs.get("username") is None
        assert conn_kwargs.get("password") is None

        settings = RedisMessagingSettings()
        with warnings.catch_warnings(record=True) as caught_settings:
            warnings.simplefilter("always")
            client_from_settings = async_redis_from_settings(
                settings, username="opt_user", password="secret"
            )
        assert any("credential_provider" in str(w.message) for w in caught_settings)
        settings_conn = client_from_settings.connection_pool.connection_kwargs
        assert (
            settings_conn["credential_provider"] is conn_kwargs["credential_provider"]
        )
        assert settings_conn.get("username") is None
        assert settings_conn.get("password") is None

        await client.aclose()
        await client_from_settings.aclose()
    finally:
        _client_cache.clear()
        clear_cached_credential_providers()


async def test_explicit_credential_provider_overrides_factory_setting(
    monkeypatch: pytest.MonkeyPatch,
):
    _client_cache.clear()
    clear_cached_credential_providers()
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        f"{__name__}:_create_provider_for_dotted_import",
    )
    explicit_provider = UsernamePasswordCredentialProvider("explicit", "token")
    try:
        client = get_async_redis_client(credential_provider=explicit_provider)
        assert (
            client.connection_pool.connection_kwargs["credential_provider"]
            is explicit_provider
        )
        await client.aclose()
    finally:
        _client_cache.clear()
        clear_cached_credential_providers()


async def test_async_redis_from_settings_explicit_none_provider_behavior():
    _client_cache.clear()
    clear_cached_credential_providers()
    try:
        with_factory = RedisMessagingSettings(
            credential_provider_factory=f"{__name__}:_create_provider_for_dotted_import"
        )
        client = async_redis_from_settings(with_factory, credential_provider=None)
        conn_kwargs = client.connection_pool.connection_kwargs
        assert isinstance(conn_kwargs["credential_provider"], CredentialProvider)
        assert conn_kwargs.get("username") is None
        assert conn_kwargs.get("password") is None
        await client.aclose()

        without_factory = RedisMessagingSettings()
        with patch("prefect_redis.client.Redis", wraps=Redis) as wrapped_redis:
            plain_client = async_redis_from_settings(
                without_factory, credential_provider=None
            )
        assert "credential_provider" not in wrapped_redis.call_args.kwargs
        plain_kwargs = plain_client.connection_pool.connection_kwargs
        assert plain_kwargs.get("credential_provider") is None
        assert plain_kwargs["username"] == "default"
        await plain_client.aclose()
    finally:
        _client_cache.clear()
        clear_cached_credential_providers()


async def test_unset_credential_provider_preserves_static_credentials():
    _client_cache.clear()
    clear_cached_credential_providers()
    try:
        client = get_async_redis_client(username="user_a", password="pass_a")
        conn_kwargs = client.connection_pool.connection_kwargs
        assert conn_kwargs["username"] == "user_a"
        assert conn_kwargs["password"] == "pass_a"
        assert conn_kwargs.get("credential_provider") is None
        await client.aclose()
    finally:
        _client_cache.clear()


async def test_credential_provider_authenticates_and_rotates_on_live_redis(
    redis: Redis,
    monkeypatch: pytest.MonkeyPatch,
):
    global _LIVE_FACTORY_CALLS
    acl_user = f"prefect_test_{uuid.uuid4().hex[:12]}"
    pass1 = f"pass1_{uuid.uuid4().hex[:8]}"
    pass2 = f"pass2_{uuid.uuid4().hex[:8]}"
    provider = _RecordingCredentialProvider(acl_user, pass1)

    await redis.execute_command(
        "ACL", "SETUSER", acl_user, "on", f">{pass1}", "~*", "+@all"
    )
    await clear_cached_clients()
    clear_cached_credential_providers()
    _LIVE_PROVIDERS[:] = [provider]
    _LIVE_FACTORY_CALLS = 0
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        f"{__name__}:_create_live_provider",
    )

    clients: list[Redis] = []
    try:
        client1 = get_async_redis_client()
        clients.append(client1)
        assert await client1.ping() is True
        assert _LIVE_FACTORY_CALLS == 1
        assert provider.calls >= 1

        # Rotate credentials in Redis and on the provider, then force new client
        # after clear_cached_clients(); provider must stay cached and supply pass2.
        await redis.execute_command(
            "ACL", "SETUSER", acl_user, "resetpass", f">{pass2}"
        )
        provider.password = pass2
        await clear_cached_clients()

        client2 = get_async_redis_client()
        clients.append(client2)
        assert await client2.ping() is True
        assert _LIVE_FACTORY_CALLS == 1
        assert provider.calls >= 2
    finally:
        for client in clients:
            await client.aclose()
        await clear_cached_clients()
        clear_cached_credential_providers()
        _LIVE_PROVIDERS.clear()
        _LIVE_FACTORY_CALLS = 0
        await redis.execute_command("ACL", "DELUSER", acl_user)


async def test_credential_provider_url_with_bad_credentials_authenticates_on_live_redis(
    redis: Redis,
    isolated_redis_db_number: int,
    monkeypatch: pytest.MonkeyPatch,
):
    acl_user = f"prefect_url_{uuid.uuid4().hex[:12]}"
    acl_pass = f"url_pass_{uuid.uuid4().hex[:8]}"
    provider = _RecordingCredentialProvider(acl_user, acl_pass)

    await redis.execute_command(
        "ACL", "SETUSER", acl_user, "on", f">{acl_pass}", "~*", "+@all"
    )
    await clear_cached_clients()
    clear_cached_credential_providers()
    _LIVE_PROVIDERS[:] = [provider]
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_CREDENTIAL_PROVIDER_FACTORY",
        f"{__name__}:_create_live_provider",
    )
    monkeypatch.setenv(
        "PREFECT_REDIS_MESSAGING_URL",
        f"redis://wrong_user:wrong_secret@localhost:6379/{isolated_redis_db_number}",
    )

    client: Redis | None = None
    try:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            client = get_async_redis_client()
        assert any("credential_provider" in str(w.message) for w in caught)
        assert all("wrong_secret" not in str(w.message) for w in caught)
        assert await client.ping() is True
        assert provider.calls >= 1
    finally:
        if client is not None:
            await client.aclose()
        await clear_cached_clients()
        clear_cached_credential_providers()
        _LIVE_PROVIDERS.clear()
        await redis.execute_command("ACL", "DELUSER", acl_user)
