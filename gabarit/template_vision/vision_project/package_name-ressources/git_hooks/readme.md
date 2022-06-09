{% if dvc_config_ok is true %}The files present in this folder are **examples** of GIT hooks with DVC.  

They are installed (simple copy/paste) by default at package initialization via `make init-local-env`.  

Thus, they are not the hooks directly used by GIT. If you want to modify the hooks, you have to edit those presents in `.git/hooks/`.
{% endif %}
