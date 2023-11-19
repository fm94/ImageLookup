from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class GT:
    """ a simple class that is used for evaluation and holding the GT data """
    # path of the ground truth
    path: str | Path
    # a dictionnary that holds the pairs: <query file name>: <source file name>
    gt: dict[str, str] = field(default_factory=dict)
    
    def __post_init__(self):
        self._load_gt()
    
    def _load_gt(self) -> None:
        """ load the gt from disk """
        data_dict = {}
        with open(self.path, 'r') as file:
            next(file)
            for line in file:
                content = line.replace('\n', '').split(' ')
                if len(content) == 2 and content[1] != '':
                    query, source = content
                else:
                    query = content[0]
                    source = '-'
                data_dict[query] = source
        self.gt = data_dict
        
    def evaluate(self, results: dict[str, str]) -> tuple[float, int]:
        """ compute the number of matches of a given result """
        common_keys = set(self.gt.keys()) & set(results.keys())
        corrects = sum(1 for key in common_keys if self.gt[key] == results[key][0])
        return corrects/len(self.gt)*100, corrects