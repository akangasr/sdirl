
from sdirl.rl.utils import Transition, Path, PathTreeIterator

class TestPathTreeIterator():

    def test_should_return_all_paths_in_tree(self):
        t1 = Transition(1, 5, 0)
        t2 = Transition(2, 6, 9)
        t3 = Transition(3, 7, 8)
        t4 = Transition(4, 8, 7)
        t5 = Transition(5, 9, 6)
        t6 = Transition(6, 0, 5)
        paths = {
            "A" : ((t1, "B"),
                   (t2, "C"),
                   (t3, "D")),
            "B" : ((t4, "C"),),
            "C" : ((t5, "B"),),
            "D" : ((t6, "A"),)
            }
        root = "A"
        maxlen = 3
        it = PathTreeIterator(root, paths, maxlen)
        all_paths = list()
        for p in it:
            all_paths.append(p)
        expected_paths = [
            Path([t1, t4, t5]),
            Path([t2, t5, t4]),
            Path([t3, t6, t1]),
            Path([t3, t6, t2]),
            Path([t3, t6, t3])
            ]
        assert len(all_paths) == len(expected_paths)
        for ep in expected_paths:
            assert ep in all_paths
