# Release Notes
## PyMC3 3.0rc1 (September 4th, 2016)
PyMC3 has seen some excellent development in the past year, we've ironed out some bugs in the NUTS sampler,
and we've enhanced the examples and documentation. 

We have model evaluation tools such as Leave one out Cross Validation, WAIC, DIC which make this a much more robust tool. Since criticism of models is important. 

One major feature worth highlighting is the addition of Automatic Differentiation Variational Inference. 
Taku Yoshioka did a lot of work on ADVI in PyMC3, including the mini-batch implementation as well as the sampling from the variational posterior. We’d also like to the thank the Stan guys (specifically Alp Kucukelbir and Daniel Lee) for deriving ADVI and teaching us about it.

We on the PyMC3 core team would like to thank everyone for contributing and now feel that this is ready for the big time. We look forward to hearing about all the cool stuff you use PyMC3 for, and further development will continue. 
## Contributors

A Kuz <for.akuz@gmail.com>
A. Flaxman <abie@alum.mit.edu>
Abraham Flaxman <abie@alum.mit.edu>
Alexey Goldin <alexey.goldin@gmail.com>
Anand Patil <anand.prabhakar.patil@gmail.com>
Andrea Zonca <code@andreazonca.com>
Andreas Klostermann <andreasklostermann@googlemail.com>
Andres Asensio Ramos <thomas.wiecki@gmail.com>
Andrew Clegg <andrew.clegg@pearson.com>
Anjum48 <thomas.wiecki@gmail.com>
AustinRochford <arochford@monetate.com>
Benjamin Edwards <bedwards@cs.unm.edu>
Boris Avdeev <borisaqua@gmail.com>
Brian Naughton <briannaughton@gmail.com>
Byron Smith <thomas.wiecki@gmail.com>
Chad Heyne <chadheyne@gmail.com>
Chris Fonnesbeck <chris.fonnesbeck@vanderbilt.edu>
Colin <thomas.wiecki@gmail.com>
Corey Farwell <coreyf@rwell.org>
David Huard <david.huard@gmail.com>
David Huard <huardda@angus.meteo.mcgill.ca>
David Stück <dstuck@users.noreply.github.com>
DeliciousHair <mshepit@gmail.com>
Dustin Tran <thomas.wiecki@gmail.com>
Eigenblutwurst <Hannes.Bathke@gmx.net>
Gideon Wulfsohn <gideon.wulfsohn@gmail.com>
Gil Raphaelli <g@raphaelli.com>
Gogs <gogitservice@gmail.com>
Ilan Man 
Imri Sofer <imrisofer@gmail.com>
Jake Biesinger <jake.biesinger@gmail.com>
James Webber <jamestwebber@gmail.com>
John McDonnell <john.v.mcdonnell@gmail.com>
John Salvatier <jsalvatier@gmail.com>
Jordi Diaz <thomas.wiecki@gmail.com>
Jordi Warmenhoven <jordi.warmenhoven@gmail.com>
Karlson Pfannschmidt <kiudee@mail.uni-paderborn.de>
Kyle Bishop <citizenphnix@gmail.com>
Kyle Meyer <kyle@kyleam.com>
Lin Xiao <thomas.wiecki@gmail.com>
Mack Sweeney <mackenzie.sweeney@gmail.com>
Matthew Emmett <memmett@unc.edu>
Maxim <thomas.wiecki@gmail.com>
Michael Gallaspy <gallaspy.michael@gmail.com>
Nick <nalourie@example.com>
Osvaldo Martin <aloctavodia@gmail.com>
Patricio Benavente <patbenavente@gmail.com>
Peadar Coyle (springcoil) <peadarcoyle@googlemail.com>
Raymond Roberts <thomas.wiecki@gmail.com>
Rodrigo Benenson <rodrigo.benenson@gmail.com>
Sergei Lebedev <superbobry@gmail.com>
Skipper Seabold <chris.fonnesbeck@vanderbilt.edu>
Taku Yoshioka <thomas.wiecki@gmail.com>
The Gitter Badger <badger@gitter.im>
Thomas Kluyver <takowl@gmail.com>
Thomas Wiecki <thomas.wiecki@gmail.com>
Tobias Knuth <mail@tobiasknuth.de>
Volodymyr <thomas.wiecki@gmail.com>
Volodymyr Kazantsev <thomas.wiecki@gmail.com>
Wes McKinney <wesmckinn@gmail.com>
Zach Ploskey <zploskey@gmail.com>
akuz <for.akuz@gmail.com>
aloctavodia <aloctavodia@gmail.com>
brandon willard <brandonwillard@gmail.com>
dstuck <dstuck88@gmail.com>
ingmarschuster <ingmar.schuster.linguistics@gmail.com>
jan-matthis <mail@jan-matthis.de>
jason <JasonTam22@gmailcom>
jonsedar <jon.sedar@applied.ai>
kiudee <quietdeath@gmail.com>
maahnman <github@mm.maahn.de>
macgyver <neil.rabinowitz@merton.ox.ac.uk>
mwibrow <mwibrow@gmail.com>
olafSmits <o.smits@gmail.com>
paul sorenson <paul@metrak.com>
redst4r <redst4r@web.de>
santon <steven.anton@idanalytics.com>
sgenoud <stevegenoud+github@gmail.com>
stonebig <stonebig>
taku-y <taku.yoshioka.4096@gmail.com>
tyarkoni <tyarkoni@gmail.com>
x2apps <x2apps@yahoo.com>
zenourn <daniel@zeno.co.nz>
## PyMC3 3.0b (June 16th, 2015)

Probabilistic programming allows for flexible specification of Bayesian statistical models in code. PyMC3 is a new, open-source probabilistic programmer framework with an intuitive, readable and concise, yet powerful, syntax that is close to the natural notation statisticians use to describe models. It features next-generation fitting techniques, such as the No U-Turn Sampler, that allow fitting complex models with thousands of parameters without specialized knowledge of fitting algorithms.

PyMC3 has recently seen rapid development. With the addition of two new major features: automatic transforms and missing value imputation, PyMC3 has become ready for wider use. PyMC3 is now refined enough that adding features is easy, so we don't expect adding features in the future will require drastic changes. It has also become user friendly enough for a broader audience. Automatic transformations mean NUTS and find_MAP work with less effort, and friendly error messages mean its easy to diagnose problems with your model.

Thus, Thomas, Chris and I are pleased to announce that PyMC3 is now in Beta.

### Highlights
* Transforms now automatically applied to constrained distributions
* Transforms now specified with a `transform=` argument on Distributions. `model.TransformedVar` is gone.
* Transparent missing value imputation support added with MaskedArrays or pandas.DataFrame NaNs.
* Bad default values now ignored
* Profile theano functions using `model.profile(model.logpt)`

### Contributors since 3.0a
* A. Flaxman <abie@alum.mit.edu>
* Andrea Zonca <code@andreazonca.com>
* Andreas Klostermann <andreasklostermann@googlemail.com>
* Andrew Clegg <andrew.clegg@pearson.com>
* AustinRochford <arochford@monetate.com>
* Benjamin Edwards <bedwards@cs.unm.edu>
* Brian Naughton <briannaughton@gmail.com>
* Chad Heyne <chadheyne@gmail.com>
* Chris Fonnesbeck <chris.fonnesbeck@vanderbilt.edu>
* Corey Farwell <coreyf@rwell.org>
* John Salvatier <jsalvatier@gmail.com>
* Karlson Pfannschmidt <quietdeath@gmail.com>
* Kyle Bishop <citizenphnix@gmail.com>
* Kyle Meyer <kyle@kyleam.com>
* Mack Sweeney <mackenzie.sweeney@gmail.com>
* Osvaldo Martin <aloctavodia@gmail.com>
* Raymond Roberts <rayvroberts@gmail.com>
* Rodrigo Benenson <rodrigo.benenson@gmail.com>
* Thomas Wiecki <thomas.wiecki@gmail.com>
* Zach Ploskey <zploskey@gmail.com>
* maahnman <github@mm.maahn.de>
* paul sorenson <paul@metrak.com>
* zenourn <daniel@zeno.co.nz>
