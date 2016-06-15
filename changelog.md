# PyMC3 Change Log

## [Unreleased](https://github.com/pymc-devs/pymc3/tree/HEAD)

[Full Changelog](https://github.com/pymc-devs/pymc3/compare/v3.0beta...HEAD)

**Closed issues:**

- Issues/confusion using multidimensional arrays in a model [\#1176](https://github.com/pymc-devs/pymc3/issues/1176)
- Referencing the package? missing in DOCS [\#1165](https://github.com/pymc-devs/pymc3/issues/1165)
- Code gets stuck when defining model [\#1162](https://github.com/pymc-devs/pymc3/issues/1162)
- Memory usage with sparse matrices [\#1157](https://github.com/pymc-devs/pymc3/issues/1157)
- ADVI returns NaN with many features [\#1147](https://github.com/pymc-devs/pymc3/issues/1147)
- Having trouble with custom distribution [\#1145](https://github.com/pymc-devs/pymc3/issues/1145)
- Hierarchical ADVI example broken [\#1119](https://github.com/pymc-devs/pymc3/issues/1119)
- hierarchical example needs annotation [\#1117](https://github.com/pymc-devs/pymc3/issues/1117)
- Feature Request: Option between Theano & Tensorflow Backend \(eventually\) [\#1115](https://github.com/pymc-devs/pymc3/issues/1115)
- Two questions on sum of distributions/convolution [\#1108](https://github.com/pymc-devs/pymc3/issues/1108)
- Shape of LKJ [\#1106](https://github.com/pymc-devs/pymc3/issues/1106)
- Still errors with Cox model [\#1105](https://github.com/pymc-devs/pymc3/issues/1105)
- minor bug in link fn for glm.families.Poisson [\#1100](https://github.com/pymc-devs/pymc3/issues/1100)
- Categorical distribution breaks when specified correctly [\#1098](https://github.com/pymc-devs/pymc3/issues/1098)
- ElemwiseCategoricalStep is broken [\#1096](https://github.com/pymc-devs/pymc3/issues/1096)

**Merged pull requests:**

- Fixing link to tutorial in readme [\#1185](https://github.com/pymc-devs/pymc3/pull/1185) ([jan-matthis](https://github.com/jan-matthis))
- Proposal: changed minibatch from iterable of generators to generator of iterable [\#1182](https://github.com/pymc-devs/pymc3/pull/1182) ([JasonTam](https://github.com/JasonTam))
- Stan model arma [\#1181](https://github.com/pymc-devs/pymc3/pull/1181) ([springcoil](https://github.com/springcoil))
- remove unused imports from examples [\#1179](https://github.com/pymc-devs/pymc3/pull/1179) ([gwulfs](https://github.com/gwulfs))
- Added random number seed to test\_sampling [\#1178](https://github.com/pymc-devs/pymc3/pull/1178) ([fonnesbeck](https://github.com/fonnesbeck))
- Added support information to README [\#1177](https://github.com/pymc-devs/pymc3/pull/1177) ([fonnesbeck](https://github.com/fonnesbeck))
- Updated Metropolis and BinaryMetropolis docstrings [\#1173](https://github.com/pymc-devs/pymc3/pull/1173) ([fonnesbeck](https://github.com/fonnesbeck))
- ADVI reports average ELBO [\#1160](https://github.com/pymc-devs/pymc3/pull/1160) ([fonnesbeck](https://github.com/fonnesbeck))
- Expanded BEST example [\#1158](https://github.com/pymc-devs/pymc3/pull/1158) ([fonnesbeck](https://github.com/fonnesbeck))
- Fix invlogit [\#1156](https://github.com/pymc-devs/pymc3/pull/1156) ([taku-y](https://github.com/taku-y))
- Advi minibatch shared [\#1152](https://github.com/pymc-devs/pymc3/pull/1152) ([taku-y](https://github.com/taku-y))
- fix error when passing axes [\#1151](https://github.com/pymc-devs/pymc3/pull/1151) ([aloctavodia](https://github.com/aloctavodia))
- Added logit and inverse logit functions [\#1149](https://github.com/pymc-devs/pymc3/pull/1149) ([fonnesbeck](https://github.com/fonnesbeck))
- Restores ElemwiseCategorical competence [\#1148](https://github.com/pymc-devs/pymc3/pull/1148) ([fonnesbeck](https://github.com/fonnesbeck))
- Automatic scaling of categorical and multinomial probabilities [\#1143](https://github.com/pymc-devs/pymc3/pull/1143) ([fonnesbeck](https://github.com/fonnesbeck))
- Used SkipTest decorator to bypass test\_examples [\#1142](https://github.com/pymc-devs/pymc3/pull/1142) ([fonnesbeck](https://github.com/fonnesbeck))
- Converted all theano.tensor T aliases to tt [\#1141](https://github.com/pymc-devs/pymc3/pull/1141) ([fonnesbeck](https://github.com/fonnesbeck))
- Remove tests of examples for CI [\#1140](https://github.com/pymc-devs/pymc3/pull/1140) ([fonnesbeck](https://github.com/fonnesbeck))
- Fixed NameError in find\_MAP [\#1138](https://github.com/pymc-devs/pymc3/pull/1138) ([fonnesbeck](https://github.com/fonnesbeck))
- added vim swap files to .gitignore, removed disp duplication in starting.py [\#1137](https://github.com/pymc-devs/pymc3/pull/1137) ([tknuth](https://github.com/tknuth))
- WIP Enforce sum-to-one for Categorical and Multinomial probabilities [\#1135](https://github.com/pymc-devs/pymc3/pull/1135) ([fonnesbeck](https://github.com/fonnesbeck))
- Allow for different starting values for each chain when njobs\>1 [\#1134](https://github.com/pymc-devs/pymc3/pull/1134) ([fonnesbeck](https://github.com/fonnesbeck))
- fixed typo: removed duplicate words [\#1133](https://github.com/pymc-devs/pymc3/pull/1133) ([tknuth](https://github.com/tknuth))
- DOCS: Made slight changes to the changelog [\#1129](https://github.com/pymc-devs/pymc3/pull/1129) ([springcoil](https://github.com/springcoil))
- BlockedStep \_\_new\_\_ arguments for unpickling to fix parallel sampling. [\#1127](https://github.com/pymc-devs/pymc3/pull/1127) ([benavente](https://github.com/benavente))
- Eliminated VisibleDeprecationWarning from summary [\#1123](https://github.com/pymc-devs/pymc3/pull/1123) ([fonnesbeck](https://github.com/fonnesbeck))
- Convert docs to sphinx [\#1122](https://github.com/pymc-devs/pymc3/pull/1122) ([twiecki](https://github.com/twiecki))
- Sampling from variational posterior [\#1121](https://github.com/pymc-devs/pymc3/pull/1121) ([taku-y](https://github.com/taku-y))
- prior plots for traceplot [\#1120](https://github.com/pymc-devs/pymc3/pull/1120) ([fonnesbeck](https://github.com/fonnesbeck))
- Notebook cleanup [\#1118](https://github.com/pymc-devs/pymc3/pull/1118) ([fonnesbeck](https://github.com/fonnesbeck))
- Replaced seed with random\_seed in advi [\#1114](https://github.com/pymc-devs/pymc3/pull/1114) ([fonnesbeck](https://github.com/fonnesbeck))
- New options added to plot\_posterior [\#1113](https://github.com/pymc-devs/pymc3/pull/1113) ([aloctavodia](https://github.com/aloctavodia))
- Added plot\_posterior to survival example [\#1111](https://github.com/pymc-devs/pymc3/pull/1111) ([fonnesbeck](https://github.com/fonnesbeck))
- Updated BEST example to use plot\_posterior [\#1110](https://github.com/pymc-devs/pymc3/pull/1110) ([fonnesbeck](https://github.com/fonnesbeck))
- Remove --process-dependency-links in favor of requrements.txt [\#1109](https://github.com/pymc-devs/pymc3/pull/1109) ([fonnesbeck](https://github.com/fonnesbeck))
- plot\_posterior\(\) added to plots.py [\#1107](https://github.com/pymc-devs/pymc3/pull/1107) ([twiecki](https://github.com/twiecki))
- bugfix for glm poisson plus new example poisson reg [\#1102](https://github.com/pymc-devs/pymc3/pull/1102) ([jonsedar](https://github.com/jonsedar))
- Shutting down ElemwiseCategorical until fixed [\#1097](https://github.com/pymc-devs/pymc3/pull/1097) ([fonnesbeck](https://github.com/fonnesbeck))

## [v3.0beta](https://github.com/pymc-devs/pymc3/tree/v3.0beta) (2015-06-18)
[Full Changelog](https://github.com/pymc-devs/pymc3/compare/v3.0alpha...v3.0beta)

## [v3.0alpha](https://github.com/pymc-devs/pymc3/tree/v3.0alpha) (2013-05-05)


\* *This Change Log was automatically generated by [github_changelog_generator](https://github.com/skywinder/Github-Changelog-Generator)*
