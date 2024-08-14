import unittest
from datetime import datetime

from parea.schemas import TestCase, TestCaseCollection, TestCaseResult


class TestTestCaseCollection(unittest.TestCase):
    def setUp(self):
        self.collection = TestCaseCollection(
            id=1,
            name="Test Collection",
            created_at="2023-05-24",
            last_updated_at="2023-05-24",
            test_cases={
                1: TestCase(
                    id=1,
                    test_case_collection_id=0,
                    inputs={"messages": "Answer this question", "context": "Short context"},
                    target="Certainly!",
                    tags=["important", "easy"],
                ),
                2: TestCase(
                    id=2,
                    test_case_collection_id=0,
                    inputs={"messages": "Solve this problem", "context": "Long context with more than 100 characters" * 3},
                    target="Sure, I can help!",
                    tags=["important", "hard"],
                ),
                3: TestCase(
                    id=3,
                    test_case_collection_id=0,
                    inputs={"messages": "Explain this concept", "word_count": "75"},
                    target="Of course!",
                    tags=["medium"],
                ),
                4: TestCase(
                    id=4,
                    test_case_collection_id=0,
                    inputs={"messages": "Analyze this data", "data_size": "1000", "timestamp": "2023-05-25T10:30:00"},
                    target="Here's the analysis:",
                    tags=["data", "analysis", "important"],
                ),
                5: TestCase(
                    id=5,
                    test_case_collection_id=0,
                    inputs={"messages": "Summarize this article", "word_count": "500", "language": "English"},
                    target="Here's a summary:",
                    tags=["summary", "medium", "language"],
                ),
                6: TestCase(
                    id=6,
                    test_case_collection_id=0,
                    inputs={"messages": "Translate this sentence", "source_language": "English", "target_language": "French"},
                    target="Voici la traduction:",
                    tags=["translation", "language", "easy"],
                ),
            },
        )

    def test_testcases_property(self):
        """Test the 'testcases' property of TestCaseCollection works like a list."""
        self.assertEqual(len(self.collection.testcases), 6)
        self.assertIsInstance(self.collection.testcases[0], TestCase)

    def test_getitem_testcases(self):
        """Test the indexing and slicing capabilities of TestCaseCollection testcases property."""
        self.assertEqual(self.collection.testcases[0].id, 1)  # Test single index access
        self.assertEqual(len(self.collection.testcases[:2]), 2)  # Test slicing
        self.assertIsInstance(self.collection.testcases[:2], TestCaseResult)  # Slicing should return TestCaseResult
        self.assertIsInstance(self.collection.testcases[:2], list)  # but still a list

    def test_getitem(self):
        """Test the indexing and slicing capabilities of TestCaseCollection."""
        self.assertEqual(self.collection[0].id, 1)  # Test single index access
        self.assertEqual(len(self.collection[:2]), 2)  # Test slicing
        self.assertIsInstance(self.collection[:2], TestCaseResult)  # Slicing should return TestCaseResult
        self.assertIsInstance(self.collection[:2], list)  # but still a list

    def test_filter_by_id(self):
        """Test filtering TestCases by their id."""
        result = self.collection.filter_testcases(id=2)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 2)

    def test_filter_by_target(self):
        """Test filtering TestCases by their target field."""
        result = self.collection.filter_testcases(target="Certainly!")
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)

    def test_filter_by_inputs_basic(self):
        """Test basic filtering of TestCases by their inputs."""
        result = self.collection.filter_testcases(inputs={"messages": "Answer this question"})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)

    def test_filter_by_tags_any(self):
        """
        Test filtering TestCases by tags using 'any' match (default behavior).

        This should return TestCases that have at least one of the specified tags.
        """
        result = self.collection.filter_testcases(tags=["important", "medium"])
        self.assertEqual(len(result), 5)
        result = self.collection.filter_testcases(tags=["medium"])
        self.assertEqual(len(result), 2)

    def test_filter_by_tags_all(self):
        """
        Test filtering TestCases by tags using 'all' match.

        This should return TestCases that have all the specified tags.
        """
        result = self.collection.filter_testcases(tags={"match": "all", "tags": ["important", "hard"]})
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 2)

    def test_filter_by_inputs_advanced(self):
        """
        Test advanced filtering of TestCases by their inputs using custom functions.

        This test demonstrates how to use lambda functions to create complex filtering conditions.
        """
        result = self.collection.filter_testcases(
            inputs=[
                ("messages", lambda x: "question" in x.lower()),  # Check if 'question' is in the message
                ("context", lambda x: len(x) < 50),  # Check if context is less than 50 characters
            ]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)

        # Filter by word count range
        result = self.collection.filter_testcases(
            inputs=[
                ("word_count", lambda x: x.isdigit() and 50 < int(x) < 100),
                # Check if word count is between 50 and 100
            ]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 3)

        # Filter by timestamp
        result = self.collection.filter_testcases(
            inputs=[
                ("timestamp", lambda x: datetime.fromisoformat(x) > datetime(2023, 5, 25)),  # Check if timestamp is after May 25, 2023
            ]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 4)

        # Filter by multiple input fields
        result = self.collection.filter_testcases(
            inputs=[
                ("messages", lambda x: "translate" in x.lower()),  # Check if 'translate' is in the message
                ("source_language", lambda x: x == "English"),  # Check if source language is English
                ("target_language", lambda x: x == "French"),  # Check if target language is French
            ]
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 6)

    def test_combined_filters(self):
        """
        Test combining multiple filters.

        This test demonstrates how to use input filtering and tag filtering together.
        """
        result = self.collection.filter_testcases(inputs=[("messages", lambda x: x.startswith("Answer"))], tags=["important"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 1)

        # Combine input filtering, tag filtering, and id filtering
        result = self.collection.filter_testcases(id=4, inputs=[("data_size", lambda x: int(x) > 500)], tags=["data", "important"])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 4)

        # Combine multiple input filters with tag filtering
        result = self.collection.filter_testcases(
            inputs=[
                ("messages", lambda x: "summarize" in x.lower()),
                ("word_count", lambda x: int(x) > 400),
            ],
            tags={"match": "all", "tags": ["medium", "language"]},
        )
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].id, 5)

        # Complex combination of filters
        result = self.collection.filter_testcases(
            inputs=[
                ("messages", lambda x: len(x.split()) > 2),  # Messages with more than 2 words
                ("word_count", lambda x: int(x) > 50 if x.isdigit() else True),  # Word count > 50 if present
            ],
            tags=["important", "medium", "language"],  # Match any of these tags
            target=lambda x: len(x) < 20,  # Target response less than 20 characters
        )
        self.assertEqual(len(result), 2)
        self.assertIn(result[0].id, [3, 5])
        self.assertIn(result[1].id, [3, 5])

    def test_no_match(self):
        """Test the behavior when no TestCases match the filter criteria."""
        result = self.collection.filter_testcases(id=999)  # No TestCase has this id
        self.assertEqual(len(result), 0)

    def test_get_all_test_case_inputs(self):
        """Test the get_all_test_case_inputs method."""
        result = self.collection.filter_testcases(inputs={"messages": "Answer this question"}).get_all_test_case_inputs()
        self.assertEqual(len(result), 1)
        self.assertIn("messages", result[0])
        self.assertEqual(result[0]["messages"], "Answer this question")

    def test_get_all_test_case_targets(self):
        """Test the get_all_test_case_targets method."""
        result = self.collection.filter_testcases(
            inputs=[
                ("messages", lambda x: "question" in x.lower()),
                ("context", lambda x: len(x) < 50),
            ]
        ).get_all_test_case_targets()
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], "Certainly!")

    def test_get_all_test_inputs_and_targets_dict(self):
        """Test the get_all_test_inputs_and_targets_dict method."""
        result = self.collection.filter_testcases(tags=["important"]).get_all_test_inputs_and_targets_dict()
        self.assertEqual(len(result), 3)
        for item in result:
            self.assertIn("inputs", item)
            self.assertIn("target", item)

    def test_chaining_methods(self):
        """Test chaining multiple methods."""
        result = self.collection.filter_testcases(tags=["important"])[1:3].get_all_test_case_targets()
        self.assertEqual(len(result), 2)
        self.assertIn("Sure, I can help!", result)
        self.assertIn("Here's the analysis:", result)


if __name__ == "__main__":
    unittest.main()
