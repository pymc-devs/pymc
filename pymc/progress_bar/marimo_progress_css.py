#   Copyright 2026 - present The PyMC Developers
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
DEFAULT_CSS = """
<style>
.pymc-progress-table {
    font-family: ui-monospace, SFMono-Regular, SF Mono, Menlo, Consolas, Liberation Mono, monospace;
    font-size: 13px;
    border-collapse: collapse;
}
.pymc-progress-table th {
    text-align: left;
    padding: 4px 10px;
    font-weight: 500;
    color: #666;
    border-bottom: 1px solid #ddd;
}
.pymc-progress-table td {
    padding: 6px 10px;
    white-space: nowrap;
}
.pymc-progress-bar-container {
    width: 120px;
    height: 14px;
    background-color: #e0e0e0;
    border-radius: 3px;
    overflow: hidden;
}
.pymc-progress-bar {
    height: 100%;
    background-color: #1f77b4;
    transition: width 0.1s ease-out;
}
.pymc-progress-bar.failing {
    background-color: #d62728;
}
.pymc-progress-bar.finished {
    background-color: #1f77b4;
}
@media (prefers-color-scheme: dark) {
    .pymc-progress-table th {
        color: #aaa;
        border-bottom-color: #444;
    }
    .pymc-progress-bar-container {
        background-color: #444;
    }
}
</style>
"""
