# Author @abcheng.
from matching.base_matcher import BaseMatcher
from matching.match import Match
from typing import List
import uuid
import os

class NaiveMatcher(BaseMatcher):
    """
    A naive matcher that puts two songs together without checking for compatibility.
    """

    def generate_matches(self, sample_directory: str, max_size: int = -1, out_path: str = "match_out", sort = "unsorted") -> List[Match]:
        """
        Naively pairs songs together and outputs it into a json list, as well as returns it.\\
        @param sample_directory: directory where all the songs exists.\\
        @param max_size: maximum size of the resulting json list. -1 means unchanged.\\
        @param out_path: output_path of the resulting jsonl file. if an out_dir was created, we write to that directory.\\
        @returns a list of Match objects, which represents matches.
        """
        result = []
        all_songs = []
        for entry_name in os.listdir(sample_directory):
            if entry_name.split(".")[-1] == "mp3": # ensures that file suffix is an mp3
                all_songs.append(entry_name)
        # create pairwise entries with no repeats.
        # this also assumes there are no duplicates.
        for i in range (len(all_songs)):
            for j in range(i+1, len(all_songs)):
                # create a match with default score of 0.0
                match = Match(uuid.uuid4(), 
                              sample_directory, 
                              [all_songs[i], all_songs[j]])
                # and then add to the list
                result.append(match)
        # truncate if necessary
        if max_size > 0:
            result = result[:max_size]
        # write to out path
        if self.out_dir:
            out_path = f"{self.out_dir}/{out_path}"
        with open(f"{out_path}.jsonl", "w") as file:
            for item in result:
                file.write(item.to_json() + '\n')
        # and also return the result
        return result



if __name__ == "__main__":
    sample_directory = os.path.abspath('data/sample')
    matcher = NaiveMatcher()
    matcher.generate_matches(sample_directory)

