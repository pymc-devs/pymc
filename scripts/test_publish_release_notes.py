#   Copyright 2024 - present The PyMC Developers
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
import re

from publish_release_notes_to_discourse import format_release_content


class TestFormatReleaseContent:
    def test_pr_links_are_formatted(self):
        """Test that PR links are formatted correctly in the release body."""
        # Realistic release body from v5.26.0
        release_body = """<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Major Changes 🛠
* Bump PyTensor dependency and drop support for NumPy <2.0, and Python 3.10 by @ricardoV94 in https://github.com/pymc-devs/pymc/pull/7910
  * Functions that try to infer inputs of the graph fail with more than one input. This includes `Model.compile_fn`, `gradient`, `jacobian` and `hessian_diag`.
  * `Model.compile_logp` now expects all model variables as input, even when only a subset of the logp terms is requested.
  * Many pytensor functions from moved from `pytensor.graph.basic` to `pytensor.graph.traversal`, including `ancestors`, `graph_inputs`, and toposort related functions.
* Remove deprecated `noise` parameter for GPs by @williambdean in https://github.com/pymc-devs/pymc/pull/7886

### New Features 🎉
* Implement `logcdf` for `CensoredRV` by @asifzubair in https://github.com/pymc-devs/pymc/pull/7884
* Derive logprob for Split operation by @ricardoV94 in https://github.com/pymc-devs/pymc/pull/7875
### Bugfixes 🪲
* Fix bug in mixture logprob inference with `None` indices by @asifzubair in https://github.com/pymc-devs/pymc/pull/7877
### Documentation 📖
* Add model_to_mermaid to docs by @williambdean in https://github.com/pymc-devs/pymc/pull/7868
* Use rst code-block over code:: by @williambdean in https://github.com/pymc-devs/pymc/pull/7882
### Maintenance 🔧
* Show more digits of step size in progress_bar by @ricardoV94 in https://github.com/pymc-devs/pymc/pull/7870
* Allow for specification of 'var_names' in 'mock_sample' by @tomicapretto in https://github.com/pymc-devs/pymc/pull/7906

## New Contributors
* @asifzubair made their first contribution in https://github.com/pymc-devs/pymc/pull/7871

**Full Changelog**: https://github.com/pymc-devs/pymc/compare/v5.25.1...v5.26.0"""

        config = {
            "RELEASE_TAG": "v5.26.0",
            "REPO_NAME": "pymc-devs/pymc",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc/releases/tag/v5.26.0",
        }

        title, content = format_release_content(config)

        # Check that the title is correct
        assert title == "🚀 Release v5.26.0"

        # Check that PR links in pymc-devs/pymc are formatted correctly
        assert "[#7910](https://github.com/pymc-devs/pymc/pull/7910)" in content
        assert "[#7886](https://github.com/pymc-devs/pymc/pull/7886)" in content
        assert "[#7884](https://github.com/pymc-devs/pymc/pull/7884)" in content
        assert "[#7875](https://github.com/pymc-devs/pymc/pull/7875)" in content
        assert "[#7877](https://github.com/pymc-devs/pymc/pull/7877)" in content
        assert "[#7868](https://github.com/pymc-devs/pymc/pull/7868)" in content
        assert "[#7882](https://github.com/pymc-devs/pymc/pull/7882)" in content
        assert "[#7870](https://github.com/pymc-devs/pymc/pull/7870)" in content
        assert "[#7906](https://github.com/pymc-devs/pymc/pull/7906)" in content
        assert "[#7871](https://github.com/pymc-devs/pymc/pull/7871)" in content

        # Check that the raw PR link format is not present
        assert (
            "https://github.com/pymc-devs/pymc/pull/7910" in content
        )  # it's still in the formatted link
        # But NOT as a standalone link (which would appear without the [#xxx](...) wrapper)
        # We can verify this by checking the pattern doesn't match the raw format
        # Find raw PR links (not inside markdown link syntax)
        raw_pr_links = re.findall(
            r"(?<!\()\bhttps://github\.com/pymc-devs/pymc/pull/\d+(?!\))", content
        )
        assert len(raw_pr_links) == 0, f"Found raw PR links: {raw_pr_links}"

        # Check that other links remain unchanged (e.g., the Full Changelog link)
        assert (
            "**Full Changelog**: https://github.com/pymc-devs/pymc/compare/v5.25.1...v5.26.0"
            in content
        )

    def test_non_pymc_links_unchanged(self):
        """Test that PR links from other repositories are not affected."""
        release_body = """Some changes:
* Feature from external repo in https://github.com/other-org/other-repo/pull/123
* Our feature in https://github.com/pymc-devs/pymc/pull/456
"""

        config = {
            "RELEASE_TAG": "v1.0.0",
            "REPO_NAME": "pymc-devs/pymc",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc/releases/tag/v1.0.0",
        }

        _title, content = format_release_content(config)

        # Check that pymc-devs/pymc PR link is formatted
        assert "[#456](https://github.com/pymc-devs/pymc/pull/456)" in content

        # Check that other repo PR link is unchanged
        assert "https://github.com/other-org/other-repo/pull/123" in content
        assert "[#123](https://github.com/other-org/other-repo/pull/123)" not in content

    def test_release_structure(self):
        """Test that the overall release structure is correct."""
        config = {
            "RELEASE_TAG": "v1.2.3",
            "REPO_NAME": "pymc-devs/pymc",
            "RELEASE_BODY": "Test body with PR https://github.com/pymc-devs/pymc/pull/999",
            "RELEASE_URL": "https://github.com/pymc-devs/pymc/releases/tag/v1.2.3",
        }

        title, content = format_release_content(config)

        assert title == "🚀 Release v1.2.3"
        assert "A new release of **pymc** is now available!" in content
        assert "**Version:** `v1.2.3`" in content
        assert "**Repository:** [pymc-devs/pymc](https://github.com/pymc-devs/pymc)" in content
        assert "**Release Page:** https://github.com/pymc-devs/pymc/releases/tag/v1.2.3" in content
        assert "[#999](https://github.com/pymc-devs/pymc/pull/999)" in content

    def test_already_formatted_links_not_double_formatted(self):
        """Test that already-formatted PR links are not double-formatted."""
        release_body = """Some changes:
* Already formatted: [#123](https://github.com/pymc-devs/pymc/pull/123)
* Raw link: https://github.com/pymc-devs/pymc/pull/456
"""

        config = {
            "RELEASE_TAG": "v1.0.0",
            "REPO_NAME": "pymc-devs/pymc",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc/releases/tag/v1.0.0",
        }

        _title, content = format_release_content(config)

        # Already formatted link should remain unchanged
        assert "[#123](https://github.com/pymc-devs/pymc/pull/123)" in content
        # Should NOT be double-formatted
        assert "[#123]([#123](https://github.com/pymc-devs/pymc/pull/123))" not in content

        # Raw link should be formatted
        assert "[#456](https://github.com/pymc-devs/pymc/pull/456)" in content

    def test_user_mentions_converted_to_links(self):
        """Test that user mentions are converted to GitHub profile links."""
        release_body = """<!-- Release notes generated using configuration in .github/release.yml at main -->

## What's Changed
### Major Changes 🛠
* Bump PyTensor dependency by @ricardoV94 in https://github.com/pymc-devs/pymc/pull/7910
* Remove deprecated parameter by @williambdean in https://github.com/pymc-devs/pymc/pull/7886

### New Features 🎉
* Implement feature by @asifzubair in https://github.com/pymc-devs/pymc/pull/7884

### Bugfixes 🪲
* Fix bug by @tomicapretto in https://github.com/pymc-devs/pymc/pull/7877

## New Contributors
* @asifzubair made their first contribution in https://github.com/pymc-devs/pymc/pull/7871
* @user-with-dashes contributed in https://github.com/pymc-devs/pymc/pull/7872
* @another_user123 helped out in https://github.com/pymc-devs/pymc/pull/7873

**Full Changelog**: https://github.com/pymc-devs/pymc/compare/v5.25.1...v5.26.0"""

        config = {
            "RELEASE_TAG": "v5.26.0",
            "REPO_NAME": "pymc-devs/pymc",
            "RELEASE_BODY": release_body,
            "RELEASE_URL": "https://github.com/pymc-devs/pymc/releases/tag/v5.26.0",
        }

        _title, content = format_release_content(config)

        # Check that user mentions are converted to links
        assert "[@ricardoV94](https://github.com/ricardoV94)" in content
        assert "[@williambdean](https://github.com/williambdean)" in content
        assert "[@asifzubair](https://github.com/asifzubair)" in content
        assert "[@tomicapretto](https://github.com/tomicapretto)" in content
        assert "[@user-with-dashes](https://github.com/user-with-dashes)" in content
        assert "[@another_user123](https://github.com/another_user123)" in content

        # Check that PR links are still formatted correctly
        assert "[#7910](https://github.com/pymc-devs/pymc/pull/7910)" in content
        assert "[#7886](https://github.com/pymc-devs/pymc/pull/7886)" in content
