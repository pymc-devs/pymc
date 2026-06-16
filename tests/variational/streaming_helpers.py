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
"""Shared helpers for the streaming-dataset tests."""


def chunked_factory(data, size):
    """Return a zero-arg factory that replays ``data`` in ``size``-row chunks.

    A ``DataLoader`` restarts its source once per epoch, so the source has to be
    re-readable. This returns a *factory* (a zero-arg callable) that produces a
    fresh generator each call, the way an out-of-core source like
    ``parquet_source`` does; a bare generator would be one-shot and could not be
    replayed. The
    final chunk may hold fewer than ``size`` rows -- the loader re-batches the
    stream to ``batch_size`` regardless -- so this also exercises the loader's
    re-batching across uneven source blocks.
    """

    def factory():
        for i in range(0, len(data), size):
            yield data[i : i + size]

    return factory
