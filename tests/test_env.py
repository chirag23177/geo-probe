"""The .env loader. Small surface, but it decides which credential gets used."""

from __future__ import annotations

import os

import pytest

from geo_probe.providers.base import ProviderError, load_dotenv, require_env


@pytest.fixture
def clean_env(monkeypatch):
    for name in ("ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY", "GEO_PROBE_TEST_KEY"):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


def _write(tmp_path, body: str):
    p = tmp_path / ".env"
    p.write_text(body, encoding="utf-8")
    return p


def test_missing_file_is_not_an_error(tmp_path, clean_env):
    assert load_dotenv(tmp_path / "nope.env") == []


def test_loads_simple_pairs(tmp_path, clean_env):
    path = _write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-abc\nPERPLEXITY_API_KEY=pplx-xyz\n")
    assert sorted(load_dotenv(path)) == ["ANTHROPIC_API_KEY", "PERPLEXITY_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abc"
    assert require_env("PERPLEXITY_API_KEY") == "pplx-xyz"


def test_exported_variable_wins_over_the_file(tmp_path, clean_env):
    """A key you just exported must not be silently replaced by a stale file."""
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    path = _write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-from-file\n")
    assert load_dotenv(path) == []
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-shell"


def test_override_is_opt_in(tmp_path, clean_env):
    clean_env.setenv("ANTHROPIC_API_KEY", "sk-ant-from-shell")
    path = _write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-from-file\n")
    assert load_dotenv(path, override=True) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-from-file"


def test_an_empty_exported_value_is_treated_as_unset(tmp_path, clean_env):
    """An empty variable is a footgun, not a credential -- let the file fill it."""
    clean_env.setenv("ANTHROPIC_API_KEY", "")
    path = _write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-real\n")
    assert load_dotenv(path) == ["ANTHROPIC_API_KEY"]
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-real"


def test_skips_comments_and_blank_lines(tmp_path, clean_env):
    path = _write(
        tmp_path,
        "# a comment\n"
        "\n"
        "   \n"
        "ANTHROPIC_API_KEY=sk-ant-abc\n"
        "  # indented comment\n",
    )
    assert load_dotenv(path) == ["ANTHROPIC_API_KEY"]


def test_accepts_export_prefix_and_surrounding_whitespace(tmp_path, clean_env):
    path = _write(tmp_path, "export  ANTHROPIC_API_KEY =  sk-ant-abc  \n")
    load_dotenv(path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abc"


def test_strips_matching_quotes_only(tmp_path, clean_env):
    path = _write(
        tmp_path,
        'ANTHROPIC_API_KEY="sk-ant-abc"\n'
        "PERPLEXITY_API_KEY='pplx-xyz'\n"
        'GEO_PROBE_TEST_KEY="mismatched\'\n',
    )
    load_dotenv(path)
    assert os.environ["ANTHROPIC_API_KEY"] == "sk-ant-abc"
    assert os.environ["PERPLEXITY_API_KEY"] == "pplx-xyz"
    assert os.environ["GEO_PROBE_TEST_KEY"] == "\"mismatched'"


def test_hash_inside_a_value_is_kept(tmp_path, clean_env):
    """No inline-comment stripping: an API key is likelier to contain '#' than
    a trailing comment is."""
    path = _write(tmp_path, "GEO_PROBE_TEST_KEY=abc#def\n")
    load_dotenv(path)
    assert os.environ["GEO_PROBE_TEST_KEY"] == "abc#def"


def test_equals_inside_a_value_is_kept(tmp_path, clean_env):
    path = _write(tmp_path, "GEO_PROBE_TEST_KEY=abc=def=ghi\n")
    load_dotenv(path)
    assert os.environ["GEO_PROBE_TEST_KEY"] == "abc=def=ghi"


def test_malformed_line_raises_with_a_line_number(tmp_path, clean_env):
    path = _write(tmp_path, "ANTHROPIC_API_KEY=sk-ant-abc\nthis line has no equals sign\n")
    with pytest.raises(ValueError, match=r":2: expected KEY=value"):
        load_dotenv(path)


def test_empty_key_raises(tmp_path, clean_env):
    path = _write(tmp_path, "=value\n")
    with pytest.raises(ValueError, match="empty key"):
        load_dotenv(path)


def test_gold_files_are_distinct_so_samples_are_never_merged(tmp_path, clean_env):
    """Fix 8: the recovered sample must not overwrite or merge into the original."""
    from geo_probe.extract import GOLD_RECOVERED, GOLD_TEMPLATE

    assert GOLD_TEMPLATE != GOLD_RECOVERED


def test_require_env_error_names_both_ways_to_set_it(clean_env):
    with pytest.raises(ProviderError) as exc:
        require_env("ANTHROPIC_API_KEY")
    message = str(exc.value)
    assert ".env" in message
    assert "export" in message
