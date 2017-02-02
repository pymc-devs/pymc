from .models import simple_2model
from .helpers import SeededTest
from pymc3 import Normal
from pymc3.sampling import transform_start_particles
import numpy as np


class TestTransformStartConsistent(SeededTest):
    def setUp(self):
        super(TestTransformStartConsistent, self).setUp()
        _, self.model, = simple_2model()
        with self.model:
            Normal('z', mu=np.array([1,2]), sd=np.array([1,2]), shape=2)

    def generate_start_points(self, size, vs=None):
        if vs is None:
            vs = self.model.vars
        return {v.name: v.distribution.random(size=size) for v in vs}

    def assert_correct(self, start, nparticles, njobs, vs=None):
        self.assertEqual(len(start), njobs, "njobs mismatched")
        first_dims = np.asarray([np.asarray(v).shape[0] for i in start for k, v in i.iteritems()])
        if nparticles is not None:
            np.testing.assert_array_equal(first_dims == nparticles, True, "Dimensions not consistent")

    def _test_start_input(self, dict_length, list_length, nparticles, njobs, vs=None):
        start = self.generate_start_points(dict_length)
        if list_length > 0:
            start = [start] * list_length
        test_this = transform_start_particles(start, nparticles, njobs, self.model)
        print start
        print test_this
        self.assert_correct(test_this, nparticles, njobs, vs)

    def _test_start_input_mismatched_particles(self, dict_lengths, list_length, nparticles, njobs, vs=None):
        startx = self.generate_start_points(dict_lengths[0])
        starty = self.generate_start_points(dict_lengths[1])
        startz = self.generate_start_points(dict_lengths[2])
        start = {}
        for s in [startx, starty, startz]:
            start.update(s)
        if list_length > 0:
            start = [start] * njobs

        with self.assertRaises(TypeError) as context:
            transform_start_particles(start, nparticles, njobs, self.model)



    def test_single_job_one_particle_list_input(self):
        self._test_start_input(1, 1, 1, 1)

    def test_single_job_one_particle_dict_input(self):
        self._test_start_input(1, 0, 1, 1)

    def test_single_job_no_particles_list_input(self):
        self._test_start_input(1, 1, None, 1)

    def test_single_job_no_particles_dict_input(self):
        self._test_start_input(1, 0, None, 1)

    def test_single_job_many_particles_list_input(self):
        self._test_start_input(3, 1, 3, 1)

    def test_single_job_many_particles_dict_input(self):
        self._test_start_input(3, 0, 3, 1)

    def test_many_jobs_no_particles_list_input(self):
        self._test_start_input(1, 3, None, 3)

    def test_many_jobs_no_particles_dict_input(self):
        self._test_start_input(3, 0, None, 3)

    def test_many_jobs_one_particle_list_input(self):
        self._test_start_input(1, 3, 1, 3)

    def test_many_jobs_one_particle_dict_input(self):
        self._test_start_input(3, 0, 1, 3)

    def test_mismatched_nparticles_single_job_many_particles_list_input(self):
        self._test_start_input_mismatched_particles([5, 6, 7], 1, 5, 1)

    def test_mismatched_nparticles_dict_input(self):
        self._test_start_input_mismatched_particles([5, 6, 7], 0, 5, 1)

    def test_duplicate_nparticles_dict(self):
        self._test_start_input(1, 0, 10, 1)
        self._test_start_input(1, 0, 10, 10)
        self._test_start_input(1, 0, 1, 10)
