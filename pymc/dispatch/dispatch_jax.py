#   Copyright 2024 The PyMC Developers
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
import jax

from pytensor.link.jax.dispatch import jax_funcify

from pymc.distributions.continuous import TruncatedNormalRV


@jax_funcify.register(TruncatedNormalRV)
def jax_funcify_TruncatedNormalRV(op, **kwargs):
    def trunc_normal_fn(key, size, mu, sigma, lower, upper):
        rng_key = key["jax_state"]
        rng_key, sampling_key = jax.random.split(rng_key, 2)
        key["jax_state"] = rng_key

        truncnorm = jax.nn.initializers.truncated_normal(sigma, lower=lower, upper=upper)

        return key, truncnorm(key["jax_state"], size) + mu

    return trunc_normal_fn
