"""Tests for named custom provider and 'main' alias resolution in auxiliary_client."""

import os
from unittest.mock import patch, MagicMock

import pytest


@pytest.fixture(autouse=True)
def _isolate(tmp_path, monkeypatch):
    """Redirect HERMES_HOME and clear module caches."""
    hermes_home = tmp_path / ".hermes"
    hermes_home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(hermes_home))
    # Write a minimal config so load_config doesn't fail
    (hermes_home / "config.yaml").write_text("model:\n  default: test-model\n")


def _write_config(tmp_path, config_dict):
    """Write a config.yaml to the test HERMES_HOME."""
    import yaml
    config_path = tmp_path / ".hermes" / "config.yaml"
    config_path.write_text(yaml.dump(config_dict))


class TestNormalizeVisionProvider:
    """_normalize_vision_provider should resolve 'main' to actual main provider."""

    def test_main_resolves_to_named_custom(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "custom:beans"},
            "custom_providers": [{"name": "beans", "base_url": "http://localhost/v1"}],
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "custom:beans"

    def test_main_resolves_to_openrouter(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "anthropic/claude-sonnet-4", "provider": "openrouter"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "openrouter"

    def test_main_resolves_to_deepseek(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "deepseek-chat", "provider": "deepseek"},
        })
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "deepseek"

    def test_main_falls_back_to_custom_when_no_provider(self, tmp_path):
        _write_config(tmp_path, {"model": {"default": "gpt-4o"}})
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("main") == "custom"

    def test_bare_provider_name_unchanged(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("beans") == "beans"
        assert _normalize_vision_provider("deepseek") == "deepseek"

    def test_custom_colon_named_provider_preserved(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("custom:beans") == "beans"

    def test_codex_alias_still_works(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("codex") == "openai-codex"

    def test_auto_unchanged(self):
        from agent.auxiliary_client import _normalize_vision_provider
        assert _normalize_vision_provider("auto") == "auto"
        assert _normalize_vision_provider(None) == "auto"


class TestResolveProviderClientMainAlias:
    """resolve_provider_client('main', ...) should resolve to actual main provider."""

    def test_main_resolves_to_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "override-model")
        assert client is not None
        assert model == "override-model"
        assert "beans.local" in str(client.base_url)

    def test_main_with_custom_colon_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "my-model", "provider": "custom:beans"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("main", "test")
        assert client is not None
        assert "beans.local" in str(client.base_url)

    def test_main_resolves_github_copilot_alias(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "gpt-5.4", "provider": "github-copilot"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "ghu_test_token",
                "base_url": "https://api.githubcopilot.com",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "gpt-5.4")

        assert client is not None
        assert model == "gpt-5.4"
        assert mock_openai.called


class TestResolveProviderClientNamedCustom:
    """resolve_provider_client should resolve named custom providers directly."""

    def test_named_custom_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("beans", "my-model")
        assert client is not None
        assert model == "my-model"
        assert "beans.local" in str(client.base_url)

    def test_named_custom_provider_default_model(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "main-model"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1", "api_key": "k"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("beans")
        assert client is not None
        # Should use _read_main_model() fallback
        assert model == "main-model"

    def test_named_custom_no_api_key_uses_fallback(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test"},
            "custom_providers": [
                {"name": "local", "base_url": "http://localhost:8080/v1"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        client, model = resolve_provider_client("local", "test")
        assert client is not None
        # no-key-required should be used

    def test_nonexistent_named_custom_falls_through(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test"},
            "custom_providers": [
                {"name": "beans", "base_url": "http://beans.local/v1"},
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        # "coffee" doesn't exist in custom_providers
        client, model = resolve_provider_client("coffee", "test")
        assert client is None


class TestResolveProviderClientModelNormalization:
    """Direct-provider auxiliary routing should normalize models like main runtime."""

    def test_matching_native_prefix_is_stripped_for_main_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("main", "zai/glm-5.1")

        assert client is not None
        assert model == "glm-5.1"

    def test_non_matching_prefix_is_preserved_for_direct_provider(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client("zai", "google/gemini-2.5-pro")

        assert client is not None
        assert model == "google/gemini-2.5-pro"

    def test_aggregator_vendor_slug_is_preserved(self, monkeypatch):
        monkeypatch.setenv("OPENROUTER_API_KEY", "or-key")
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client

            client, model = resolve_provider_client(
                "openrouter", "anthropic/claude-sonnet-4.6"
            )

        assert client is not None
        assert model == "anthropic/claude-sonnet-4.6"


class TestResolveVisionProviderClientModelNormalization:
    """Vision auto-routing should reuse the same provider-specific normalization."""

    def test_vision_auto_strips_matching_main_provider_prefix(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "zai/glm-5.1", "provider": "zai"},
        })
        with (
            patch("agent.auxiliary_client._read_nous_auth", return_value=None),
            patch("hermes_cli.auth.resolve_api_key_provider_credentials", return_value={
                "api_key": "glm-key",
                "base_url": "https://api.z.ai/api/paas/v4",
            }),
            patch("agent.auxiliary_client.OpenAI") as mock_openai,
        ):
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client()

        assert provider == "zai"
        assert client is not None
        assert model == "glm-5v-turbo"  # zai has dedicated vision model in _PROVIDER_VISION_MODELS


class TestVisionPathApiMode:
    """Vision path should propagate api_mode to _get_cached_client."""

    def test_explicit_provider_passes_api_mode(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"default": "test-model"},
            "auxiliary": {"vision": {"api_mode": "chat_completions"}},
        })
        with patch("agent.auxiliary_client._get_cached_client") as mock_gcc:
            mock_gcc.return_value = (MagicMock(), "test-model")
            from agent.auxiliary_client import resolve_vision_provider_client

            provider, client, model = resolve_vision_provider_client(provider="deepseek")

        mock_gcc.assert_called_once()
        _, kwargs = mock_gcc.call_args
        assert kwargs.get("api_mode") == "chat_completions"


class TestProvidersDictApiModeAnthropicMessages:
    """Regression guard for #15033.

    Named providers declared under the ``providers:`` dict with
    ``api_mode: anthropic_messages`` must route auxiliary calls through
    the Anthropic Messages API (via AnthropicAuxiliaryClient), not
    through an OpenAI chat-completions client.

    The bug had two halves: the providers-dict branch of
    ``_get_named_custom_provider`` dropped the ``api_mode`` field, and
    ``resolve_provider_client``'s named-custom branch never read it.
    """

    def test_providers_dict_propagates_api_mode(self, tmp_path, monkeypatch):
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                },
            },
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("myrelay")
        assert entry is not None
        assert entry.get("api_mode") == "anthropic_messages"
        assert entry.get("base_url") == "https://example-relay.test/anthropic"
        assert entry.get("api_key") == "sk-test"

    def test_providers_dict_invalid_api_mode_is_dropped(self, tmp_path):
        _write_config(tmp_path, {
            "providers": {
                "weird": {
                    "name": "weird",
                    "base_url": "https://example.test",
                    "api_mode": "bogus_nonsense",
                    "default_model": "x",
                },
            },
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("weird")
        assert entry is not None
        assert "api_mode" not in entry

    def test_providers_dict_without_api_mode_is_unchanged(self, tmp_path):
        _write_config(tmp_path, {
            "providers": {
                "localchat": {
                    "name": "localchat",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "api_key": "local-key",
                    "default_model": "llama-3",
                },
            },
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("localchat")
        assert entry is not None
        assert "api_mode" not in entry

    def test_resolve_provider_client_returns_anthropic_client(self, tmp_path, monkeypatch):
        """Named custom provider with api_mode=anthropic_messages must
        route through AnthropicAuxiliaryClient."""
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                },
            },
        })
        from agent.auxiliary_client import (
            resolve_provider_client,
            AnthropicAuxiliaryClient,
            AsyncAnthropicAuxiliaryClient,
        )
        sync_client, sync_model = resolve_provider_client("myrelay", async_mode=False)
        assert isinstance(sync_client, AnthropicAuxiliaryClient), (
            f"expected AnthropicAuxiliaryClient, got {type(sync_client).__name__}"
        )
        assert sync_model == "claude-opus-4-7"

        async_client, async_model = resolve_provider_client("myrelay", async_mode=True)
        assert isinstance(async_client, AsyncAnthropicAuxiliaryClient), (
            f"expected AsyncAnthropicAuxiliaryClient, got {type(async_client).__name__}"
        )
        assert async_model == "claude-opus-4-7"

    def test_aux_task_override_routes_named_provider_to_anthropic(self, tmp_path, monkeypatch):
        """The full chain: auxiliary.<task>.provider: myrelay with
        api_mode anthropic_messages must produce an Anthropic client."""
        monkeypatch.setenv("MYRELAY_API_KEY", "sk-test")
        _write_config(tmp_path, {
            "providers": {
                "myrelay": {
                    "name": "myrelay",
                    "base_url": "https://example-relay.test/anthropic",
                    "key_env": "MYRELAY_API_KEY",
                    "api_mode": "anthropic_messages",
                    "default_model": "claude-opus-4-7",
                },
            },
            "auxiliary": {
                "compression": {
                    "provider": "myrelay",
                    "model": "claude-sonnet-4.6",
                },
            },
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        from agent.auxiliary_client import (
            get_async_text_auxiliary_client,
            get_text_auxiliary_client,
            AnthropicAuxiliaryClient,
            AsyncAnthropicAuxiliaryClient,
        )
        async_client, async_model = get_async_text_auxiliary_client("compression")
        assert isinstance(async_client, AsyncAnthropicAuxiliaryClient)
        assert async_model == "claude-sonnet-4.6"

        sync_client, sync_model = get_text_auxiliary_client("compression")
        assert isinstance(sync_client, AnthropicAuxiliaryClient)
        assert sync_model == "claude-sonnet-4.6"

    def test_provider_without_api_mode_still_uses_openai(self, tmp_path):
        """Named providers that don't declare api_mode should still go
        through the plain OpenAI-wire path (no regression)."""
        _write_config(tmp_path, {
            "providers": {
                "localchat": {
                    "name": "localchat",
                    "base_url": "http://127.0.0.1:1234/v1",
                    "api_key": "local-key",
                    "default_model": "llama-3",
                },
            },
        })
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI, AsyncOpenAI
        sync_client, _ = resolve_provider_client("localchat", async_mode=False)
        # sync returns the raw OpenAI client
        assert isinstance(sync_client, OpenAI)
        async_client, _ = resolve_provider_client("localchat", async_mode=True)
        assert isinstance(async_client, AsyncOpenAI)


class TestCustomProviderAliasCollision:
    """A user-declared custom_providers entry whose name matches a built-in
    *alias* (not a canonical provider) must win over the built-in.

    Regression guard for #15743: users who defined fallback_model pointing at
    a custom_providers entry named ``kimi`` were having requests routed to
    the built-in kimi-coding endpoint because ``_normalize_aux_provider``
    rewrote ``kimi`` → ``kimi-coding`` before the named-custom lookup.
    """

    def test_custom_named_kimi_wins_over_builtin_alias(self, tmp_path):
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
            "custom_providers": [
                {
                    "name": "kimi",
                    "base_url": "https://my-custom-kimi.example.com/v1",
                    "api_key": "my-kimi-key",
                    "models": {"my-kimi-model": {"context_length": 200000}},
                },
            ],
        })
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI
        client, model = resolve_provider_client("kimi", model="my-kimi-model", raw_codex=True)
        assert isinstance(client, OpenAI)
        assert "my-custom-kimi.example.com" in str(client.base_url)
        assert client.api_key == "my-kimi-key"
        assert model == "my-kimi-model"

    def test_bare_kimi_without_custom_still_routes_to_builtin(self, tmp_path, monkeypatch):
        """Regression guard: bare 'kimi' with no custom entry must still
        reach the built-in kimi-coding provider."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        monkeypatch.setenv("KIMI_API_KEY", "builtin-kimi-key")
        from agent.auxiliary_client import resolve_provider_client
        client, _ = resolve_provider_client("kimi", model="kimi-k2-0905-preview", raw_codex=True)
        assert client is not None
        base_url = str(client.base_url)
        # Built-in kimi-coding points at api.moonshot.ai
        assert "moonshot" in base_url or "kimi" in base_url, f"unexpected base_url {base_url!r}"

    def test_explicit_overrides_applied_on_api_key_branch(self, tmp_path, monkeypatch):
        """Explicit base_url/api_key from the caller must override the
        registered provider's defaults on the API-key branch.  Used by
        _try_activate_fallback to route a fallback through a built-in
        provider name but targeting a user-supplied endpoint."""
        _write_config(tmp_path, {
            "model": {"provider": "openrouter", "default": "anthropic/claude-sonnet-4.6"},
        })
        monkeypatch.setenv("KIMI_API_KEY", "builtin-kimi-key")
        from agent.auxiliary_client import resolve_provider_client
        from openai import OpenAI
        client, _ = resolve_provider_client(
            "kimi-coding", model="kimi-k2", raw_codex=True,
            explicit_base_url="https://override.example.com",
            explicit_api_key="override-key",
        )
        assert isinstance(client, OpenAI)
        assert "override.example.com" in str(client.base_url)
        assert client.api_key == "override-key"


class TestCustomProviderSslVerify:
    """ssl_verify config field disables TLS verification for self-signed endpoints.

    Regression guard for #28260 — custom_providers entries pointing at
    self-signed HTTPS servers need a way to opt out of certificate
    verification without touching global SSL settings.
    """

    def test_normalize_ssl_verify_false(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "local-llm",
            "base_url": "https://192.168.1.50:8443/v1",
            "api_key": "k",
            "ssl_verify": False,
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["ssl_verify"] is False

    def test_normalize_ssl_verify_true_is_preserved(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "local-llm",
            "base_url": "https://192.168.1.50:8443/v1",
            "api_key": "k",
            "ssl_verify": True,
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["ssl_verify"] is True

    def test_normalize_ssl_verify_ca_bundle_path(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "corp-llm",
            "base_url": "https://llm.corp.internal/v1",
            "api_key": "k",
            "ssl_verify": "/etc/ssl/certs/corp-ca.pem",
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["ssl_verify"] == "/etc/ssl/certs/corp-ca.pem"

    def test_normalize_ssl_verify_absent_is_not_in_result(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "local-llm",
            "base_url": "http://localhost:11434/v1",
            "api_key": "k",
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert "ssl_verify" not in result

    def test_normalize_camelcase_ssl_verify_alias(self):
        from hermes_cli.config import _normalize_custom_provider_entry
        entry = {
            "name": "local-llm",
            "base_url": "https://192.168.1.50:8443/v1",
            "api_key": "k",
            "sslVerify": False,
        }
        result = _normalize_custom_provider_entry(entry)
        assert result is not None
        assert result["ssl_verify"] is False

    def test_get_named_custom_provider_passes_ssl_verify(self, tmp_path):
        _write_config(tmp_path, {
            "custom_providers": [
                {
                    "name": "selfsigned",
                    "base_url": "https://192.168.1.50:8443/v1",
                    "api_key": "k",
                    "ssl_verify": False,
                }
            ],
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("selfsigned")
        assert entry is not None
        assert entry.get("ssl_verify") is False

    def test_providers_dict_passes_ssl_verify(self, tmp_path):
        _write_config(tmp_path, {
            "providers": {
                "selfsigned": {
                    "name": "selfsigned",
                    "base_url": "https://192.168.1.50:8443/v1",
                    "api_key": "k",
                    "ssl_verify": False,
                }
            },
        })
        from hermes_cli.runtime_provider import _get_named_custom_provider
        entry = _get_named_custom_provider("selfsigned")
        assert entry is not None
        assert entry.get("ssl_verify") is False

    def test_resolve_provider_client_passes_http_client_when_ssl_verify_false(self, tmp_path):
        """resolve_provider_client must pass httpx.Client(verify=False) to OpenAI
        when the provider entry declares ssl_verify: false."""
        import httpx
        _write_config(tmp_path, {
            "custom_providers": [
                {
                    "name": "selfsigned",
                    "base_url": "https://192.168.1.50:8443/v1",
                    "api_key": "k",
                    "ssl_verify": False,
                }
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            client, _ = resolve_provider_client("selfsigned", "test-model")

        assert mock_openai.called
        call_kwargs = mock_openai.call_args[1]
        assert "http_client" in call_kwargs, (
            "OpenAI() was not passed an http_client — ssl_verify was not threaded through"
        )
        http_client = call_kwargs["http_client"]
        assert isinstance(http_client, httpx.Client)

    def test_resolve_provider_client_no_http_client_when_ssl_verify_absent(self, tmp_path):
        """When ssl_verify is not set, no http_client override should be injected."""
        _write_config(tmp_path, {
            "custom_providers": [
                {
                    "name": "normal",
                    "base_url": "https://api.example.com/v1",
                    "api_key": "k",
                }
            ],
        })
        with patch("agent.auxiliary_client.OpenAI") as mock_openai:
            mock_openai.return_value = MagicMock()
            from agent.auxiliary_client import resolve_provider_client
            resolve_provider_client("normal", "test-model")

        assert mock_openai.called
        call_kwargs = mock_openai.call_args[1]
        assert "http_client" not in call_kwargs
