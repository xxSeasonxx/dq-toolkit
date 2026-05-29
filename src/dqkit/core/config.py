"""Typed runtime configuration, sourced from environment or defaults."""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime configuration for dqkit.

    Values are read from the environment (prefix ``DQKIT_``) and fall back to
    the defaults below. Holding configuration in one frozen, typed object keeps
    callers from reaching for loose ``os.environ`` lookups scattered in code.

    Attributes:
        app_name: Spark application name.
        log_level: Root log level for the ``dqkit`` logger.
        zscore_threshold: ``|z|`` above which a value is flagged anomalous.
        iqr_multiplier: Tukey fence multiplier for the IQR detector.
    """

    model_config = SettingsConfigDict(env_prefix="DQKIT_", frozen=True)

    app_name: str = "dqkit"
    log_level: str = "INFO"
    zscore_threshold: float = Field(default=3.0, gt=0)
    iqr_multiplier: float = Field(default=1.5, gt=0)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Returns:
        The cached :class:`Settings` instance.
    """
    return Settings()
