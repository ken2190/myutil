import numpy as np
from numpy.testing import *
import pytest
from scipy.stats import chisquare

from utilmy.stats.hypothesis.gof import ChiSquareTest, JarqueBera


class TestChiSquare(object):
    obs, exp = [29, 19, 18, 25, 17, 10, 15, 11], [18, 18, 18, 18, 18, 18, 18, 18]

    def test_chisquaretest(self):
        """ TestChiSquare:test_chisquaretest.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        chi_test = ChiSquareTest(self.obs, self.exp)
        sci_chi_test = chisquare(self.obs, self.exp)

        assert_almost_equal(chi_test.chi_square, sci_chi_test.statistic)
        assert_almost_equal(chi_test.p_value, sci_chi_test.pvalue)

        assert not chi_test.continuity_correction
        assert chi_test.degrees_of_freedom == len(self.obs) - 1

    def test_chisquaretest_arr(self):
        """ TestChiSquare:test_chisquaretest_arr.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        chi_test = ChiSquareTest(np.array(self.obs), np.array(self.exp))
        sci_chi_test = chisquare(self.obs, self.exp)

        assert_almost_equal(chi_test.chi_square, sci_chi_test.statistic)
        assert_almost_equal(chi_test.p_value, sci_chi_test.pvalue)

        assert not chi_test.continuity_correction
        assert chi_test.degrees_of_freedom == len(self.obs) - 1

    def test_chisquaretest_continuity(self):
        """ TestChiSquare:test_chisquaretest_continuity.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        chi_test = ChiSquareTest(self.obs, self.exp, continuity=True)

        assert_almost_equal(chi_test.chi_square, 14.333333333333334)
        assert_almost_equal(chi_test.p_value, 0.045560535300404756)

        assert chi_test.continuity_correction

    def test_chisquare_no_exp(self):
        """ TestChiSquare:test_chisquare_no_exp.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        chi_test = ChiSquareTest(self.obs)
        sci_chi_test = chisquare(self.obs, self.exp)

        assert_almost_equal(chi_test.chi_square, sci_chi_test.statistic)
        assert_almost_equal(chi_test.p_value, sci_chi_test.pvalue)

    def test_chisquare_exceptions(self):
        """ TestChiSquare:test_chisquare_exceptions.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        with pytest.raises(ValueError):
            ChiSquareTest(self.obs, self.exp[:5])


class TestJarqueBera(object):

    def test_jarquebera(self):
        """ TestJarqueBera:test_jarquebera.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        pass

    def test_jarquebera_exceptions(self):
        """ TestJarqueBera:test_jarquebera_exceptions.
        Doc::
                
                    Args:
                    Returns:
                       
        """
        pass
