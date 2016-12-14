import pandas_vectors as pv
import pandas as pd
import numpy as np

import unittest

class PvTest(unittest.TestCase):
    def test_indexer(self):
        self.assertListEqual(pv.indexer('a'), ['a_x', 'a_y', 'a_z'])
        self.assertListEqual(pv.indexer(['a']), ['a_x', 'a_y', 'a_z'])
        self.assertListEqual(pv.indexer('abc'), ['abc_x', 'abc_y', 'abc_z'])
        self.assertListEqual(pv.indexer(['abc']), ['abc_x', 'abc_y', 'abc_z'])
        self.assertListEqual(pv.indexer(['abc','def']), ['abc_x', 'abc_y', 'abc_z', 'def_x', 'def_y', 'def_z'])

    def test_vectornames(self):
        pv.set_vectornames('pyr')
        self.assertListEqual(pv.indexer('a'), ['a_p', 'a_y', 'a_r'])
        pv.set_vectornames(['_l', '_m', '_n', '_o'])
        self.assertListEqual(pv.indexer('a'), ['a_l', 'a_m', 'a_n', 'a_o'])
        with pv.vectornames('xyz'):
            self.assertListEqual(pv.indexer('a'), ['a_x', 'a_y', 'a_z'])
        with pv.vectornames('xy'):
            self.assertListEqual(pv.indexer('a'), ['a_x', 'a_y'])
        self.assertListEqual(pv.indexer('a'), ['a_l', 'a_m', 'a_n', 'a_o'])


if __name__ == '__main__':
    unittest.main()
