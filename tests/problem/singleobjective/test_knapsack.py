import unittest
from unittest import mock

import numpy as np
import pytest

from jmetal.problem.singleobjective.knapsack import Knapsack


class KnapsackTestCases(unittest.TestCase):
    def test_should_constructor_create_a_non_null_object(self) -> None:
        problem = Knapsack()
        self.assertIsNotNone(problem)

    def test_should_constructor_create_a_valid_problem_with_default_settings(self) -> None:
        problem = Knapsack()
        self.assertEqual(50, problem.number_of_variables())  # Each bit is a separate variable
        self.assertEqual(1, problem.number_of_objectives())
        self.assertEqual(1, problem.number_of_constraints())
        self.assertEqual(50, problem.number_of_bits)
        self.assertEqual(1000, problem.capacity)
        self.assertIsNone(problem.profits)
        self.assertIsNone(problem.weights)

    def test_should_constructor_create_a_valid_problem_with_500_bits(self) -> None:
        problem = Knapsack(500)
        self.assertEqual(500, problem.number_of_variables())  # Each bit is a separate variable
        self.assertEqual(1, problem.number_of_objectives())
        self.assertEqual(1, problem.number_of_constraints())
        self.assertEqual(500, problem.number_of_bits)
        self.assertEqual(1000, problem.capacity)
        self.assertIsNone(problem.profits)
        self.assertIsNone(problem.weights)

    def test_should_create_solution_a_valid_binary_solution(self) -> None:
        problem = Knapsack(256)
        solution = problem.create_solution()
        # Check that the solution has a bits attribute with the correct length
        self.assertTrue(hasattr(solution, "bits"))
        self.assertEqual(256, len(solution.bits))

    def test_should_create_solution_from_file(self) -> None:
        filename = "resources/Knapsack_instances/KnapsackInstance_50_0_0.kp"

        data = (
            "50\n13629\n 865 445\n395 324\n777 626\n912 656\n431 935\n42 210 \n266 990\n989 566\n524 489\n"
            "498 454\n415 887\n941 534\n803 267\n850 64 \n311 825\n992 941\n489 562\n367 938\n598 15 \n914 96 \n"
            "930 737\n224 861\n517 409\n143 728\n289 845\n144 804\n774 685\n98 641 \n634 2  \n819 627\n257 506\n"
            "932 848\n546 889\n723 342\n830 250\n617 748\n924 334\n151 721\n318 892\n102 65 \n748 196\n76 940 \n"
            "921 582\n871 228\n701 245\n339 823\n484 991\n574 146\n104 823\n363 557"
        )
        with mock.patch(
            "jmetal.problem.singleobjective.knapsack.open", new=mock.mock_open(read_data=data)
        ):
            problem = Knapsack(from_file=True, filename=filename)
            self.assertEqual(50, problem.number_of_variables())  # Each bit is a separate variable
            self.assertEqual(1, problem.number_of_objectives())
            self.assertEqual(1, problem.number_of_constraints())
            self.assertEqual(50, problem.number_of_bits)

    def test_should_get_name_return_the_right_name(self):
        problem = Knapsack()
        self.assertEqual("Knapsack", problem.name())


class TestKnapsackInitialization:
    """Check stored samples, random-source selection, and generator state."""

    @pytest.mark.parametrize("use_generator", [True, False])
    def test_should_store_sampled_pattern_in_returned_bits(self, use_generator: bool) -> None:
        """Both random sources must populate the solution, not a temporary list."""
        problem: Knapsack = Knapsack(5)
        expected: np.ndarray = np.array([True, False, True, False, True])
        rng: mock.Mock = mock.Mock(spec=np.random.Generator)
        rng.choice.return_value = expected
        with mock.patch(
            "jmetal.problem.singleobjective.knapsack.np.random.choice", return_value=expected
        ) as global_choice:
            solution = problem.create_solution(rng if use_generator else None)

        np.testing.assert_array_equal(solution.bits, expected)
        assert solution.variables == expected.tolist()
        assert solution.number_of_variables == 5 and solution.number_of_objectives == 1
        assert solution.objectives == [0.0] and solution.constraints == []
        selected = rng.choice if use_generator else global_choice
        selected.assert_called_once_with([True, False], size=5)
        (global_choice if use_generator else rng.choice).assert_not_called()

    def test_should_preserve_seeded_generator_draw_and_state(self) -> None:
        """A supplied generator produces the expected bits and subsequent state."""
        rng: np.random.Generator = np.random.default_rng(42)
        reference: np.random.Generator = np.random.default_rng(42)
        expected: np.ndarray = reference.choice([True, False], size=50)

        solution = Knapsack().create_solution(rng)

        np.testing.assert_array_equal(solution.bits, expected)
        assert rng.bit_generator.state == reference.bit_generator.state

    def test_should_preserve_seeded_global_numpy_draw_and_state(self) -> None:
        """Global NumPy sampling matches a seeded reference and restores shared state."""
        saved = np.random.get_state()
        reference: np.random.RandomState = np.random.RandomState(42)
        expected: np.ndarray = reference.choice([True, False], size=50)
        try:
            np.random.seed(42)
            solution = Knapsack().create_solution()
            state = np.random.get_state()
        finally:
            np.random.set_state(saved)

        np.testing.assert_array_equal(solution.bits, expected)
        expected_state = reference.get_state()
        assert state[0] == expected_state[0] and state[2:] == expected_state[2:]
        np.testing.assert_array_equal(state[1], expected_state[1])

    @pytest.mark.parametrize("use_generator", [True, False])
    def test_should_create_empty_vector_without_changing_sampling_call(
        self, use_generator: bool
    ) -> None:
        """The full setter supports the constructor's empty binary vector."""
        rng: mock.Mock = mock.Mock(spec=np.random.Generator)
        rng.choice.return_value = np.array([], dtype=bool)
        with mock.patch(
            "jmetal.problem.singleobjective.knapsack.np.random.choice",
            return_value=np.array([], dtype=bool),
        ) as global_choice:
            solution = Knapsack(0).create_solution(rng if use_generator else None)

        assert solution.variables == [] and solution.bits.size == 0
        assert solution.number_of_variables == 0
        assert solution.objectives == [0.0] and solution.constraints == []
        (rng.choice if use_generator else global_choice).assert_called_once_with(
            [True, False], size=0
        )
