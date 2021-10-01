# Main Governance Document

## The Project

The PyMC Project (The Project) is an open source software project
affiliated with the 501c3 NumFOCUS Foundation. The goal of The Project is to
develop open source software and deploy open and public websites and services
for reproducible, exploratory and interactive computing. The Software developed
by The Project is released under OSI approved open source licenses,
developed openly and hosted in public GitHub repositories under the
[pymc-devs GitHub organization](https://github.com/pymc-devs). Examples of
Project Software include the PyMC library and its documentation, etc.
The Services run by the Project consist of public websites and web-services
that are hosted at [http://docs.pymc.io](https://docs.pymc.io)

The Project is developed by a team of distributed developers, called
Contributors. Contributors are individuals who have contributed code,
documentation, designs or other work to one or more Project repositories,
or who have done significant work to empower the Community,
participating on [Discourse](https://discourse.pymc.io),
organizing [PyMCon](https://pymcon.com) or helped on other platforms and events.
Anyone can be a Contributor. Contributors can be affiliated with any legal
The foundation of Project participation is openness and transparency.

There have been over 250 Contributors to the Project, their contributions are listed in the
logs of the PyMC GitHub repositories as well as those of associated projects and venues.

The Project Community consists of all Contributors and Users of the Project.
Contributors work on behalf of and are responsible to the larger Project
Community and we strive to keep the barrier between Contributors and Users as
low as possible.

The Project is formally affiliated with the 501c3 NumFOCUS Foundation
([http://numfocus.org](http://numfocus.org)), which serves as its fiscal
sponsor, may hold project trademarks and other intellectual property, helps
manage project donations and acts as a parent legal entity. NumFOCUS is the
only legal entity that has a formal relationship with the project (see
Institutional Partners section below).

## Governance

This section outlines the governance and leadership model of The Project.

The foundations of Project governance are:

- Openness & Transparency
- Active Contribution
- Institutional Neutrality

Traditionally, Project leadership was provided by a BDFL (Chris Fonnesbeck) and
subset of Contributors, called Core Developers, whose active and consistent
contributions have been recognized by their receiving “commit rights” to the
Project GitHub repositories. In general all Project decisions are made through
consensus among the Core Developers with input from the Community. The BDFL
can, but rarely chooses to, override the Core Developers and make a final
decision on a matter.

While this approach has served us well, as the Project grows and faces more
legal and financial decisions and interacts with other institutions, we see a
need for a more formal governance and organization model.
We view this governance model as the formalization of what we are already doing,
rather than a change in direction.

## Community and Team Architecture
The PyMC community is organized in an onion-like fashion.
The tiers relevant to the project governance are listed below sorted by
increasing responsibility. Due to the onion-like structure, members of a group are
also members of all the groups listed above:

* Contributors
* Recurring Contributors
* Core Contributors
* Steering Council
* BDFL

Recurring Contributors comprise what we understand as the PyMC Team.
The Team will generally act as a single unit, except for some specific
questions where dedicated teams will prevail.
Currently there are two teams within the PyMC project,
the Developer and Documentation teams.
Team members can be part of one, some or none of these dedicated teams.

![community_diagram](docs/community_diagram.png)

Anyone working with The Project has the responsibility to personally uphold
the Code of Conduct. Core Contributors have the additional responsibility
of _enforcing_ the Code of Conduct to maintain a safe community.

## Recurring Contributors
Recurring Contributors are those individuals who contribute recurrently to the
project and can provide valuable insight on the project.
They are therefore actively consulted and can participate in the same communication
channels as Core Contributors. However, unlike Core Contributors,
Recurrent Contributors don't have voting, managing or writing rights.

In practice, this translates in participating from private team discussions
(i.e. in Slack or live meetings) but not being able to vote Steering Council
members or having commit rights on GitHub.

The Recurrent Contributor position will often be an intermediate step for people
in becoming Core Contributors once their contributions are frequent enough
and during a sustained period of time.
But it is also an important role by itself for people who want to be part of
the project but don't have the time or don't want the responsibilities that
come with being a Core Contributors.

### Recurring Contributor membership
Recurring Contributors can nominate any Contributor to participate in the
Project private communication channels (i.e. Slack public channel)
and become a Recurring Contributor.
For the nomination to go forward, it has to be ratified by the Steering Council.
For a nomination to be rejected, clear reasoning behind the decision must be
shared with the rest of the team. People whose nomination has been rejected can
be nominated at any time again in the future, three months after the previous
nomination at the earliest.

#### Current Recurring Contributors

- everyone on slack! This may be too long so it could be a good idea to
format as a table instead or to put within a `<details>` tag so
it's hidden by default while still available publicly

## Core Contributors
Core Contributors are those individuals entrusted with the development and
well being of the Project due to their frequency of quality contributions over
a sustained period of time. They are the main governing and decision body
of the Project and are therefore given voting and managing rights to the Project
services (i.e. commit rights on GitHub or moderation rights on Discourse).
The exact permissions of all Core Contributors may not be the same
and depend on their team memberships. Even if they have commit rights,
Core Contributors should still have their pull requests reviewed by at least
one other Core Contributor before merging unless prevented by a major force
reason. If overstepping, Core Contributors can also be subject to a vote
of no confidence (see below) and see their permissions revoked.

### Core Contributor membership
To become a Core Contributor, one must already be a Recurring Contributor.
Core Contributors can nominate any Recurring Contributor to become a
Core Contributor. For the nomination to go forward, it has to be
ratified by the Steering Council.
For a nomination to be rejected, clear reasoning behind the decision must be
shared with the rest of the team. People whose nomination has been rejected can
be nominated at any time again in the future, three months after the previous
nomination at the earliest.

### Current Core Contributors
<!-- a table will probably look better, but we can worry about that later -->

* Adrian Seyboldt
* Alex Andorra
* Austin Rochford
* Brandon T. Willard
* Chris Fonnesbeck
* Colin Carroll
* Eelke Spaak
* Eric Ma
* George Ho
* John Salvatier
* Junpeng Lao
* Luciano Paz
* Marco E. Gorelli
* Martina Cantaro
* Maxim Kochurov
* Meenal Jhajharia
* Michael Osthege
* Oriol Abril-Pla
* Osvaldo Martin
* Ravin Kumar
* Ricardo Vieira
* Robert P. Goldman
* Sayam Kumar
* Thomas Wiecki

## Steering Council

The Project will have a Steering Council that consists of Project Contributors
who have produced contributions that are substantial in quality and quantity,
and sustained over at least one year. The overall role of the Council is to
ensure, through working with the BDFL and taking input from the Community, the
long-term well-being of the project, both technically and as a community.

The Steering Council will have between 4 and 7 members with at least one member
per dedicated team and no more than 2 institutional members per company.

During the everyday project activities, council members participate in all
discussions, code review and other project activities as peers with all other
Contributors and the Community. In these everyday activities, Council Members
do not have any special power or privilege through their membership on the
Council. However, it is expected that because of the quality and quantity of
their contributions and their expert knowledge of the Project Software and
Services that Council Members will provide useful guidance, both technical and
in terms of project direction, to potentially less experienced contributors.

The Steering Council and its Members play a special role in certain situations.
In particular, the Council may:

- Make decisions about the overall scope, vision and direction of the
  project.
- Make decisions about strategic collaborations with other organizations or
  individuals.
- Make decisions about specific technical issues, features, bugs and pull
  requests. They are the primary mechanism of guiding the code review process
  and merging pull requests.
- Make decisions about the Services that are run by The Project and manage
  those Services for the benefit of the Project and Community.
- Make decisions when regular community discussion doesn’t produce consensus
  on an issue in a reasonable time frame.

### Current Steering Council

The current Steering Council membership comprises:

- void until we merge this and then have an election.
  The process for renewing and electing a new member is the same so it won't
  make any difference.

Institutional Contributors are indicated as `name (company)`

### Council membership

To become eligible for being a Steering Council Member an individual must be a
Core Contributor who has produced contributions that are substantial in
quality and quantity, and sustained over at least one year. Potential Council
after asking if the potential Member is interested and willing
to serve in that capacity.

When considering potential Members, the Council will look at candidates with a
comprehensive view of their contributions. This will include but is not limited
to code, code review, infrastructure work, mailing list and chat participation,
community help/building, education and outreach, design work, etc. We are
deliberately not setting arbitrary quantitative metrics (like “100 commits in
this repo”) to avoid encouraging behavior that plays to the metrics rather than
the project’s overall well-being. We want to encourage a diverse array of
backgrounds, viewpoints and talents in our team, which is why we explicitly do
not define code as the sole metric on which council membership will be
evaluated.

Council membership is assigned for a two year period, with no limit on how many
periods can be served.

Council members can renounce at any time and are
encouraged to do so if they foresee they won't be able to attend their
responsibilities for an extended interval of time.

If a Council member becomes inactive in the project for a period of six months,
they will be considered for removal from the Council. Before removal, inactive
Member will be approached by the BDFL to see if they plan on returning to
active participation. If not they will be removed immediately upon a Council
vote. If they plan on returning to active participation soon, they will be
given a grace period of six months. If they don’t return to active participation
within that time period they will be removed by vote of the Council without
further grace period. All former Council members can be considered for
membership again at any time in the future, like any other Core Contributor.
 Retired Council members will be listed on the project website, acknowledging
the period during which they were active in the Council.

The Council reserves the right to eject current Members, other than the BDFL,
if they are deemed to be actively harmful to the project’s well-being, and
attempts at communication and conflict resolution have failed.

### Private communications of the Council

Unless specifically required, all Council discussions and activities will be
public and done in collaboration and discussion with the Project Contributors
and Community. The Council will have a private mailing list that will be used
sparingly and only when a specific matter requires privacy. When private
communications and decisions are needed, the Council will do its best to
summarize those to the Community after eliding personal/private/sensitive
information that should not be posted to the public internet.

### Subcommittees

The Council can create subcommittees that provide leadership and guidance for
specific aspects of the project. Like the Council as a whole, subcommittees
should conduct their business in an open and public manner unless privacy is
specifically called for. Private subcommittee communications should happen on
the main private mailing list of the Council unless specifically called for.

Even if the BDFL does not sit on a specific subcommittee, they still retain
override authority on the subcommittee's decisions. However, it is expected that
they will appoint a delegate to oversee the subcommittee's decisions, and
explicit intervention from the BDFL will only be sought if the committee
disagrees with the delegate's decision and no resolution is possible within the
subcommittee. This is a different situation from a BDFL delegate for a specific
decision, or a recusal situation, in which the BDFL gives up their authority
to someone else in full.

### NumFOCUS Subcommittee

The Council will maintain one narrowly focused subcommittee to manage its
interactions with NumFOCUS.

- The NumFOCUS Subcommittee is comprised of 5 persons who manage project
  funding that comes through NumFOCUS. It is expected that these funds will
  be spent in a manner that is consistent with the non-profit mission of
  NumFOCUS and the direction of the Project as determined by the full
  Council.
- This Subcommittee shall NOT make decisions about the direction, scope or
  technical direction of the Project.
- This Subcommittee will have 5 members, 4 of whom will be current Council
  Members and 1 of whom will be external to the Steering Council. No more
  than 2 Subcommitee Members can report to one person through employment or
  contracting work (including the reportee, i.e. the reportee + 1 is the
  max). This avoids effective majorities resting on one person.

#### Current NumFOCUS Subcommitee
The current NumFOCUS Subcommittee consists of:

- Peadar Coyle
- Chris Fonnesbeck
- John Salvatier
- Jon Sedar
- Thomas Wiecki

## BDFL

The Project will have a BDFL (Benevolent Dictator for Life), who is currently
Chris Fonnesbeck. As Dictator, the BDFL has the authority to make all final
decisions for The Project. As Benevolent, the BDFL, in practice chooses to
defer that authority to the consensus of the community discussion channels and
the Steering Council. It is expected, and in the past has been the
case, that the BDFL will only rarely assert their final authority. Because
rarely used, we refer to BDFL’s final authority as a “special” or “overriding”
vote. When it does occur, the BDFL override typically happens in situations
where there is a deadlock in the Steering Council or if the Steering Council
asks the BDFL to make a decision on a specific matter. To ensure the
benevolence of the BDFL, The Project encourages others to fork the project if
they disagree with the overall direction the BDFL is taking. The BDFL is chair
of the Steering Council (see below) and may delegate their authority on a
particular decision or set of decisions to any other Council member at their
discretion.

The BDFL can appoint their successor, but it is expected that the Steering
Council would be consulted on this decision. If the BDFL is unable to appoint a
successor, the Steering Council will make a suggestion or suggestions to the
Main NumFOCUS Board. While the Steering Council and Main NumFOCUS Board will
work together closely on the BDFL selection process, the Main NUMFOCUS Board
will make the final decision.


## Conflict of interest

It is expected that the BDFL, Council Members and Contributors will be
employed at a wide range of companies, universities and non-profit organizations.
Because of this, it is possible that Members will have conflict of interests.
Such conflict of interests include, but are not limited to:

- Financial interests, such as investments, employment or contracting work,
  outside of The Project that may influence their work on The Project.
- Access to proprietary information of their employer that could potentially
  leak into their work with the Project.

All members of the Council, BDFL included, shall disclose to the rest of the
Council any conflict of interest they may have. Members with a conflict of
interest in a particular issue may participate in Council discussions on that
issue, but must recuse themselves from voting on the issue. If the BDFL has
recused themselves for a particular decision, they will appoint a substitute
BDFL for that decision.

## Vote of no conficence
In exceptional circumstances, Council Members as well as Core Contributors
may remove a sitting council member via a vote of no confidence.
Core contributors can also call for a vote to remove the entire council
-- in which case, Council Members do not vote.
A no-confidence vote is triggered when a Core Contributor calls for one
publicly on an appropriate project communication channel,
and two other core team members second the proposal.
The initial call for a no-confidence vote must specify which type is intended
-- whether it is targeting a single member or the council as a whole.

The vote lasts for two weeks, and the people taking part in it vary:
* If this is a single-member vote called by Core contributors,
  both Council members and Core contributors vote,
  and the vote is deemed successful if at least two thirds of voters
  express a lack of confidence.
* If this is a whole-council vote, then it was necessarily called by
  Core contributors (since Council members can’t remove the whole Council)
  and only Core contributors vote.
  The vote is deemed successful if at least two thirds of voters
  express a lack of confidence.

After voting:
* If a single-member vote on a council member succeeds, then that member is
  removed from the council and the resulting vacancy can be handled in the usual way.
* If a single-member vote on a core contributor succeeds, their permissions are
  revoked and would have to wait six months to be eligible for core contributor
  nomination again.
* If a whole-council vote succeeds, the council is dissolved and a new council election is triggered immediately.

## Ejecting Core Contributors
Core contributors can be ejected through a simple majority vote by the council. Council members vote "Yes" or "No".

Upon ejecting a core contributor the council must publish an issue ticket, or public document detailing the
* Violations
* Evidence if available
* Remediation plan (if necessary)
* Signatures majority of council members to validate correctness and accuracy

## Leaving the project
Any contributor can also voluntarily leave the project by notifying the community through a public means or by notifying the entire council. When doing so, they can add themselves to the alumni section below if desired.

People who leave the project voluntarily can rejoin at any time

## Team Organization
As stated previously, The Team will generally act as a single unit,
except for some specific questions where dedicated teams will prevail.
These dedicated teams have no difference in how they are governed.
Decisions should be reached by consensus within the team with the Steering
Council and the BDFL acting if necessary.

The dedicated teams are work units with two main objectives: better
distributing the work related to The Project, and to better showcase all the task
involved in The Project to attract more diverse Contributors.

The PyMC project currently counts with the Developer and Documentation teams.
Team members can be part of one, some or none of these dedicated teams.

### Developer Team
The focus of the developer team is the probabilistic programming library
and flagship of The Project, [PyMC](https://github.com/pymc-devs/pymc).

#### Current Developer Team

-

### Documentation Team
The focus of the documentation team is ensuring the PyMC library
is well documented, building and maintaining the infrastructure needed
for that aim and making sure there are resources to learn
Bayesian statistics with PyMC.
It is not the goal nor responsibility of the Documentation team to
write all the documentation for the PyMC library.

#### Current Documentation Team

- Abhipsha Das
- Benjamin Vincent
- Chris Fonnesbeck
- Lorenzo Toniazzi
- Martina Cantaro
- Meenal Jhajharia
- Michael Osthege
- Olga Kahn
- Oriol Abril-Pla
- Osvaldo Martin
- Raul Maldonado
- Ravin Kumar
- Sayam Kumar

### Team structure in practice

Our two teams are currently structured about GitHub centric tasks, so the
permissions on GitHub repositories is mapped to team membership and role
within the team. The team defines to which repositories the permissions
are given, the role defines the type of permissions given:

Role:
- Recurring Contributors are given triage permissions
- Core Contributors are given write permissions

Team:
* Development team members are given permissions to [pymc](https://github.com/pymc-devs/pymc) repository
* Documentation team members are given permissions to [pymc-examples](https://github.com/pymc-devs/pymc-examples)
  and [resources](https://github.com/pymc-devs/resources)
  repositories.

In addition, Council members are given admin rights to all repositories within
the [pymc-devs](https://github.com/pymc-devs) organization.

## Institutional Partners and Funding

The PyMC Core Contributors (together with the BDFL and Steering Council)
are the primary leadership for the project. No
outside institution, individual or legal entity has the ability to own,
control, usurp or influence the project other than by participating in the
Project as Contributors and Council Members. However, because institutions are
the primary funding mechanism for the project, it is important to formally
acknowledge institutional participation in the project. These are Institutional
Partners.

An Institutional Contributor is any individual Core Contributor who
contributes to the project as part of their official duties at an Institutional
Partner. Likewise, an Institutional Council Member is any Project Steering
Council Member who contributes to the project as part of their official duties
at an Institutional Partner.

With these definitions, an Institutional Partner is any recognized legal entity
in the United States or elsewhere that employs at least one Institutional
Contributor or Institutional Council Member. Institutional Partners can be
for-profit or non-profit entities.

Institutions become eligible to become an Institutional Partner by
employing individuals who actively contribute to The Project as part
of their official duties. To state this another way, the only way for
an Institutional Partner to influence the project is by actively
contributing to the open development of the project, on equal terms
with any other member of the community of Contributors and Council
Members. Merely using PyMC Software or Services in an
institutional context does not allow an entity to become an
Institutional Partner. Financial gifts do not enable an entity to
become an Institutional Partner (see Sponsors below for financial gift recognition).
Once an institution becomes eligible
for Institutional Partnership, the Steering Council must nominate and
approve the Partnership.

If an existing Institutional Partner no longer has a contributing employee,
they will be given a one-year grace period for other employees to begin
contributing.

An Institutional Partner is free to pursue funding for their work on The
Project through any legal means. This could involve a non-profit organization
raising money from private foundations and donors or a for-profit company
building proprietary products and services that leverage Project Software and
Services. Funding acquired by Institutional Partners to work on The Project is
called Institutional Funding. However, no funding obtained by an Institutional
Partner can override The Project BDFL and Steering Council. If a Partner has
funding to do PyMC work and the Council decides to not pursue that
work as a project, the Partner is free to pursue it on their own. However in
this situation, that part of the Partner’s work will not be under the
PyMC banner and cannot use the Project trademarks in a way that
suggests a formal relationship.

To acknowledge institutional contributions, there are two level of Institutional
Partners, with associated benefits:

**Tier 1** = an institution with at least one Institutional Council Member

- Acknowledged on the PyMC websites, in talks and T-shirts.
- Ability to acknowledge their own funding sources on the PyMC
  websites, in talks and T-shirts.
- Unlimited participation in the annual Institutional Partners Workshop, held
  during the (planned) annual PyMC Project Retreat. This allows the
  Institutional Partner to invite as many of their own employees and funding
  sources and collaborators as they want, even if they are not project
  Contributors or Council Members.
- Ability to influence the project through the participation of their Council
  Member.
- Council Members are invited to the bi-annual PyMC Developer Meeting.

**Tier 2** = an institution with at least one Institutional Contributor

- Same benefits as Tier 1 level Partners, but:
- Only Institutional Contributors are invited to the Institutional Partners
  Workshop and bi-annual PyMC Developer Meeting

The PyMC project currently recognizes PyMC Labs as a Tier 1 Institutional Partner,
with Thomas Wiecki and Adrian Seyboldt as their institutional contributors
and council members.

## Sponsors
Sponsors are Organizations that provide significant funding to the PyMC project
either directly or by sponsoring PyMCon.

Sponsors will be recognized by placing their logo on the PyMC website but will have
no extra benefits related to The Project. Note that PyMCon sponsors may have
extra benefits but those will be related to the conference, not the Project.

## Team Alumni

* Person + extra info if we want (examples https://mc-stan.org/about/team/ or
  https://numpy.org/doc/stable/dev/governance/people.html#emeritus-members)

<!-- We can open that to council members or core contributors if they want to list themselves here after leaving the team. -->
