from __future__ import annotations

from datetime import datetime

import polars as pl
import pytest

import iosislib.tsfn.adapters.local_sources as local_sources
from iosislib.core.graph import Graph, LocalExecutor
from iosislib.core.node import Node
from iosislib.core.tsfn import FrameSignature, TSFN, TimeAxis
from iosislib.core.utils import (
    S3Credentials,
    _s3_credentials_scope,
    current_s3_credentials,
)


def dt(minute: int) -> datetime:
    return datetime(2026, 1, 1, 0, minute)


class ProbeSource(TSFN):
    """Inputless source that records the scoped S3 credentials at execution time."""

    VERSION = "1.0.0"
    _captured: list[S3Credentials | None] = []

    def type_signature(self) -> tuple[FrameSignature, FrameSignature]:
        output = FrameSignature(
            time=TimeAxis(),
            columns=(("value", pl.Int64),),
        )
        return FrameSignature.empty(), output

    def apply(self) -> pl.LazyFrame:
        ProbeSource._captured.append(current_s3_credentials())
        return pl.DataFrame({"timestamp": [dt(0)], "value": [1]}).lazy()


@pytest.fixture(autouse=True)
def reset_probe() -> None:
    ProbeSource._captured = []


CREDENTIALS = S3Credentials(
    access_key="AKIAIOSFODNN7EXAMPLE",
    secret_key="wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY",
    session_token="SESSION",
    region="us-east-1",
)


class TestS3CredentialsValidation:
    def test_valid_credentials_with_session_token(self) -> None:
        credentials = S3Credentials(
            access_key="a",
            secret_key="b",
            session_token="t",
            region="us-east-1",
        )
        assert credentials.access_key == "a"
        assert credentials.secret_key == "b"
        assert credentials.session_token == "t"
        assert credentials.region == "us-east-1"

    def test_session_token_and_region_are_optional(self) -> None:
        credentials = S3Credentials(access_key="a", secret_key="b")
        assert credentials.session_token is None
        assert credentials.region is None

    def test_empty_access_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="access_key must be a non-empty string"):
            S3Credentials(access_key="", secret_key="b")

    def test_blank_access_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="access_key must be a non-empty string"):
            S3Credentials(access_key="   ", secret_key="b")

    def test_empty_secret_key_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="secret_key must be a non-empty string"):
            S3Credentials(access_key="a", secret_key="")

    def test_non_string_session_token_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_token must be a string or None"):
            S3Credentials(access_key="a", secret_key="b", session_token=123)

    def test_blank_session_token_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="session_token must be a string or None"):
            S3Credentials(access_key="a", secret_key="b", session_token="  ")

    def test_non_string_region_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="region must be a string or None"):
            S3Credentials(access_key="a", secret_key="b", region=123)

    def test_blank_region_is_rejected(self) -> None:
        with pytest.raises(ValueError, match="region must be a string or None"):
            S3Credentials(access_key="a", secret_key="b", region=" ")


class TestS3CredentialsScope:
    def test_default_state_has_no_credentials(self) -> None:
        assert current_s3_credentials() is None

    def test_scope_sets_and_restores_value(self) -> None:
        with _s3_credentials_scope(CREDENTIALS):
            assert current_s3_credentials() == CREDENTIALS
        assert current_s3_credentials() is None

    def test_scope_accepts_none_as_an_explicit_noop(self) -> None:
        with _s3_credentials_scope(None):
            assert current_s3_credentials() is None

    def test_scope_resolves_a_callable_provider(self) -> None:
        with _s3_credentials_scope(lambda: CREDENTIALS):
            assert current_s3_credentials() == CREDENTIALS
        assert current_s3_credentials() is None

    def test_scope_restores_outer_value_on_exception(self) -> None:
        with pytest.raises(RuntimeError, match="boom"):
            with _s3_credentials_scope(CREDENTIALS):
                raise RuntimeError("boom")
        assert current_s3_credentials() is None

    def test_scope_restores_outer_value_when_nested(self) -> None:
        inner = S3Credentials(access_key="inner", secret_key="inner")
        with _s3_credentials_scope(CREDENTIALS):
            with _s3_credentials_scope(inner):
                assert current_s3_credentials() == inner
            assert current_s3_credentials() == CREDENTIALS
        assert current_s3_credentials() is None

    def test_provider_returning_invalid_value_raises_type_error(self) -> None:
        with pytest.raises(
            TypeError, match="S3 credentials provider must return S3Credentials or None"
        ):
            with _s3_credentials_scope(lambda: object()):
                pass

    def test_scope_restores_value_after_invalid_provider(self) -> None:
        with pytest.raises(TypeError):
            with _s3_credentials_scope(lambda: object()):
                pass
        assert current_s3_credentials() is None


class TestParquetFilesystemCredentials:
    def test_s3_location_without_credentials_uses_from_uri(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pyarrow import fs as pyarrow_fs

        seen: list[tuple[object, ...]] = []
        constructed: list[dict[str, object]] = []

        class FakeFileSystem:
            @staticmethod
            def from_uri(uri: str) -> tuple[object, str]:
                seen.append((uri,))
                return ("fallback-fs", uri.removeprefix("s3://"))

        def fake_s3_file_system(**kwargs: object) -> object:
            constructed.append(kwargs)
            return object()

        monkeypatch.setattr(pyarrow_fs, "FileSystem", FakeFileSystem)
        monkeypatch.setattr(pyarrow_fs, "S3FileSystem", fake_s3_file_system)

        filesystem, path = local_sources._open_parquet_filesystem(
            "s3://market-data/prices"
        )

        assert filesystem == "fallback-fs"
        assert path == "market-data/prices"
        assert seen == [("s3://market-data/prices",)]
        assert constructed == []

    def test_s3_location_with_credentials_builds_explicit_filesystem(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pyarrow import fs as pyarrow_fs

        constructed: list[dict[str, object]] = []

        def fake_s3_file_system(**kwargs: object) -> object:
            constructed.append(kwargs)
            return "scoped-fs"

        monkeypatch.setattr(pyarrow_fs, "S3FileSystem", fake_s3_file_system)

        with _s3_credentials_scope(CREDENTIALS):
            filesystem, path = local_sources._open_parquet_filesystem(
                "s3://market-data/prices"
            )

        assert filesystem == "scoped-fs"
        assert path == "market-data/prices"
        assert constructed == [
            {
                "access_key": CREDENTIALS.access_key,
                "secret_key": CREDENTIALS.secret_key,
                "session_token": CREDENTIALS.session_token,
                "region": CREDENTIALS.region,
            }
        ]

    def test_s3_with_credentials_without_session_token(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pyarrow import fs as pyarrow_fs

        constructed: list[dict[str, object]] = []

        def fake_s3_file_system(**kwargs: object) -> object:
            constructed.append(kwargs)
            return "scoped-fs"

        monkeypatch.setattr(pyarrow_fs, "S3FileSystem", fake_s3_file_system)

        credentials = S3Credentials(access_key="a", secret_key="b")
        with _s3_credentials_scope(credentials):
            local_sources._open_parquet_filesystem("s3://bucket/key")

        assert constructed == [
            {
                "access_key": "a",
                "secret_key": "b",
                "session_token": None,
                "region": None,
            }
        ]

    def test_local_path_is_untouched_by_scoped_credentials(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from pyarrow import fs as pyarrow_fs

        created: list[tuple[object, ...]] = []

        class FakeLocalFileSystem:
            def __init__(self, **kwargs: object) -> None:
                created.append(tuple(kwargs.items()))

            def normalize_path(self, path: str) -> str:
                return path

        monkeypatch.setattr(pyarrow_fs, "LocalFileSystem", FakeLocalFileSystem)

        with _s3_credentials_scope(CREDENTIALS):
            filesystem, path = local_sources._open_parquet_filesystem("/data/prices")

        assert filesystem.normalize_path("/data/prices") == "/data/prices"
        assert path == "/data/prices"
        assert created == [()]


class TestExecutorScopesCredentials:
    def test_executor_scopes_credentials_during_execution(self) -> None:
        executor = LocalExecutor(s3_credentials=CREDENTIALS)
        Graph(Node(ProbeSource)).execute(executor=executor)

        assert ProbeSource._captured == [CREDENTIALS]
        assert current_s3_credentials() is None

    def test_executor_without_credentials_scopes_none(self) -> None:
        Graph(Node(ProbeSource)).execute(executor=LocalExecutor())

        assert ProbeSource._captured == [None]
        assert current_s3_credentials() is None

    def test_executor_resolves_provider_callable_per_execution(self) -> None:
        calls: list[int] = []

        def provider() -> S3Credentials:
            calls.append(len(calls))
            return CREDENTIALS

        executor = LocalExecutor(s3_credentials=provider)
        Graph(Node(ProbeSource)).execute(executor=executor)
        Graph(Node(ProbeSource)).execute(executor=executor)

        assert calls == [0, 1]
        assert ProbeSource._captured == [CREDENTIALS, CREDENTIALS]
        assert current_s3_credentials() is None

    def test_provider_returning_none_disables_scoped_credentials(self) -> None:
        executor = LocalExecutor(s3_credentials=lambda: None)
        Graph(Node(ProbeSource)).execute(executor=executor)

        assert ProbeSource._captured == [None]
        assert current_s3_credentials() is None

    def test_credentials_do_not_change_node_or_graph_identity(self) -> None:
        node = Node(ProbeSource)
        graph_id = Graph(node).ID

        Graph(node).execute(executor=LocalExecutor(s3_credentials=CREDENTIALS))

        assert node.ID == Node(ProbeSource).ID
        assert Graph(Node(ProbeSource)).ID == graph_id
