# Copyright (c) 2026 Trillim. Licensed under the MIT License. See LICENSE.
"""Unit tests for backend prompt-cache planning and commit helpers."""

import unittest

from trillim._prompt_cache import PromptCacheManager, PromptSnapshot


class PromptCacheManagerTests(unittest.TestCase):
    def test_restore_and_clear_reset_state(self):
        cache = PromptCacheManager()
        cache.restore(
            PromptSnapshot.create([1, 2]),
            last_cache_hit=2,
        )

        self.assertEqual(cache.token_ids, (1, 2))
        self.assertEqual(cache.last_cache_hit, 2)

        cache.restore(None)
        self.assertEqual(cache.token_ids, ())
        self.assertEqual(cache.last_cache_hit, 0)

    def test_plan_reuses_exact_cached_token_prefix(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2]))

        plan = cache.plan(PromptSnapshot.create([1, 2, 3, 4]))

        self.assertEqual(plan.request.token_ids, (1, 2, 3, 4))
        self.assertEqual(plan.delta_tokens, (3, 4))
        self.assertEqual(plan.reset_flag, 0)
        self.assertEqual(plan.cache_hit, 2)

    def test_plan_resets_when_request_is_shorter_than_cached_prefix(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2, 3]))

        plan = cache.plan(PromptSnapshot.create([1, 2]))

        self.assertEqual(plan.delta_tokens, (1, 2))
        self.assertEqual(plan.reset_flag, 1)
        self.assertEqual(plan.cache_hit, 0)

    def test_plan_resets_on_partial_prefix_match(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 9]))

        plan = cache.plan(PromptSnapshot.create([1, 2]))

        self.assertEqual(plan.delta_tokens, (1, 2))
        self.assertEqual(plan.reset_flag, 1)
        self.assertEqual(plan.cache_hit, 0)

    def test_commit_generation_keeps_backend_cached_prefix(self):
        cache = PromptCacheManager()
        plan = cache.plan(PromptSnapshot.create([1, 2]))

        cache.commit_generation(plan, generated_token_ids=[3, 4], kv_position=3)

        self.assertEqual(cache.token_ids, (1, 2, 3))
        self.assertEqual(cache.last_cache_hit, 0)

    def test_commit_generation_records_cache_hit_for_continued_request(self):
        cache = PromptCacheManager()
        cache.restore(PromptSnapshot.create([1, 2]), last_cache_hit=2)
        plan = cache.plan(PromptSnapshot.create([1, 2, 3]))

        cache.commit_generation(plan, generated_token_ids=[4], kv_position=4)

        self.assertEqual(cache.token_ids, (1, 2, 3, 4))
        self.assertEqual(cache.last_cache_hit, 2)


if __name__ == "__main__":
    unittest.main()
