import tomllib
from typing import Optional

from pydantic import BaseModel


class KeysConfig(BaseModel):
    x: str
    y: str
    group: str
    unit: str


class DataConfig(BaseModel):
    path: str
    label: str
    relabel: Optional[str] = None


class Config(BaseModel):
    map: str
    keys: KeysConfig
    data: list[DataConfig]


def load_config(filename) -> Config:
    with open(filename, "rb") as f:
        config = tomllib.load(f)

    return Config(**config)
