from typing import List, Union


class BaseSource:
    def load(self) -> List[Union[str, dict]]:
        raise NotImplementedError