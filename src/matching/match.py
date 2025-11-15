from dataclasses import dataclass
from typing import List
import json
from uuid import UUID

@dataclass
class Match:
    """
    Class for representation of a candidate matchup
    @field UUID: the unique id of the match
    @field directory: the directory where the songs exist
    @field songs: the list of songs that are included in the match
    @field score: the score of the match, if applicable.
    """
    id: UUID
    directory: str
    songs: List[str]
    score: float = 0.0

    def to_json(self):
        """
        Returns the dataclass in a json format.
        """
        as_dict = {"id": str(self.id), "directory": self.directory, "songs": self.songs, "score": self.score}
        return json.dumps(as_dict)