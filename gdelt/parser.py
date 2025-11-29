"""Stream parser for GDELT GKG files."""
from __future__ import annotations

import csv
import gzip
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional


@dataclass
class Count:
    type: str
    value: float


@dataclass
class Tone:
    tone: float
    positive: float
    negative: float
    polarity: float
    activity_density: float


@dataclass
class Location:
    name: str
    country_code: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


@dataclass
class GDELTRecord:
    datetime: datetime
    themes: List[str]
    counts: List[Count]
    tone: Tone
    persons: List[str]
    orgs: List[str]
    locations: List[Location]
    gcam: Dict[str, float]


class GDELTParser:
    """Parse raw GDELT GKG files into structured ``GDELTRecord`` objects."""

    # Column indices for the v2 GKG schema.
    IDX_DATE = 1
    IDX_COUNTS = 5
    IDX_THEMES = 6
    IDX_ENHANCED_THEMES = 7
    IDX_LOCATIONS = 8
    IDX_PERSONS = 9
    IDX_ORGS = 10
    IDX_TONE = 11
    IDX_GCAM = 15

    def parse_file(self, path: Path) -> Iterator[GDELTRecord]:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as f:
            reader = csv.reader(f, delimiter="\t")
            for row in reader:
                if len(row) <= self.IDX_TONE:
                    continue
                dt = self._parse_datetime(row[self.IDX_DATE])
                themes = self._parse_themes(row[self.IDX_THEMES], row[self.IDX_ENHANCED_THEMES])
                counts = self._parse_counts(row[self.IDX_COUNTS])
                tone = self._parse_tone(row[self.IDX_TONE])
                persons = self._parse_list(row[self.IDX_PERSONS])
                orgs = self._parse_list(row[self.IDX_ORGS])
                locations = self._parse_locations(row[self.IDX_LOCATIONS])
                gcam = self._parse_gcam(row[self.IDX_GCAM]) if len(row) > self.IDX_GCAM else {}
                yield GDELTRecord(
                    datetime=dt,
                    themes=themes,
                    counts=counts,
                    tone=tone,
                    persons=persons,
                    orgs=orgs,
                    locations=locations,
                    gcam=gcam,
                )

    @staticmethod
    def _parse_datetime(value: str) -> datetime:
        return datetime.strptime(value, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc)

    @staticmethod
    def _parse_list(value: str) -> List[str]:
        return [v for v in value.split(";") if v]

    @staticmethod
    def _parse_themes(v1: str, v2: str) -> List[str]:
        combined = set()
        combined.update([v for v in v1.split(";") if v])
        combined.update([v.split("#", 1)[0] for v in v2.split(";") if v])
        return sorted(combined)

    @staticmethod
    def _parse_counts(value: str) -> List[Count]:
        counts: List[Count] = []
        for entry in value.split(";"):
            parts = entry.split("#")
            if len(parts) < 2:
                continue
            try:
                counts.append(Count(type=parts[0], value=float(parts[1])))
            except ValueError:
                continue
        return counts

    @staticmethod
    def _parse_tone(value: str) -> Tone:
        # tone, positive, negative, polarity, activityDensity, group
        parts = value.split(",")
        floats = [float(p) for p in parts[:5]] if parts else [0.0] * 5
        while len(floats) < 5:
            floats.append(0.0)
        return Tone(*floats[:5])

    @staticmethod
    def _parse_locations(value: str) -> List[Location]:
        locations: List[Location] = []
        for entry in value.split(";"):
            parts = entry.split("#")
            if len(parts) < 4:
                continue
            name = parts[1]
            country_code = parts[3]
            latitude = float(parts[4]) if len(parts) > 4 and parts[4] else None
            longitude = float(parts[5]) if len(parts) > 5 and parts[5] else None
            locations.append(
                Location(
                    name=name,
                    country_code=country_code,
                    latitude=latitude,
                    longitude=longitude,
                )
            )
        return locations

    @staticmethod
    def _parse_gcam(value: str) -> Dict[str, float]:
        metrics: Dict[str, float] = {}
        for item in value.split(","):
            if not item:
                continue
            if ":" not in item:
                continue
            key, val = item.split(":", 1)
            try:
                metrics[key] = float(val)
            except ValueError:
                continue
        return metrics

    def parse_files(self, paths: Iterable[Path]) -> Iterator[GDELTRecord]:
        for path in paths:
            yield from self.parse_file(path)
