# Copyright 2021-2026 The pyCMF Developers. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

'''Maturity markers for methods that are still under development.'''

import warnings


class ExperimentalWarning(UserWarning):
    '''Issued when a method that has not been validated yet is instantiated.'''


def experimental(name, note=''):
    '''Warn that ``name`` is experimental.  Call it from ``__init__``.'''
    msg = ('%s is experimental: it is not covered by the validation suite and '
           'its interface may change.' % name)
    if note:
        msg += ' ' + note
    warnings.warn(msg, ExperimentalWarning, stacklevel=3)
