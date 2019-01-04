from .models import simple_model


class TestProfile:
    def setup_method(self):
        _, self.model, _ = simple_model()

    def test_profile_model(self):
        assert self.model.profile(self.model.logpt).fct_call_time > 0

    def test_profile_variable(self):
        assert self.model.profile(self.model.vars[0].logpt).fct_call_time > 0

    def test_profile_count(self):
        count = 1005
        assert self.model.profile(self.model.logpt, n=count).fct_callcount == count
