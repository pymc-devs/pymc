{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

{% if objtype == "class" %}
.. autoclass:: {{ objname }}

   .. rubric:: Methods

   .. autosummary::
      :toctree: classmethods

      {{ objname }}.dist
{% else %}
.. autofunction:: {{ objname }}
{% endif %}
