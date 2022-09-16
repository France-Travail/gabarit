# TODO list


### Overall

- Add other templates ? Such as unsupervised templates.
- Fix all TODOs
- Add a note in the README.md about torch packages management (cf. https://stackoverflow.com/questions/57689387/equivalent-for-find-links-in-setup-py)
- Fix all linters

### All templates

- Review how some models' parameters are managed. For example, the models' learning rate should probably be an __init__ argument.
- Are we checking all parameters the same way ?
- Review how the `get_classes_from_proba` and `inverse_transform` functions work! It is not very understandable!
- Add used preprocessing as models' attributes ?
- Remove / fix some `# type: ignore`.


### Template - NLP

- Installation error with `python setup.py develop` when requirements.txt is not called first -> error: requests 2.28.0 is installed but requests<2.25.1,>=2.23.0 is required by {'words-n-fun'}
- Shouldn't transformers be saved into `XXX-data` folder ? Like detectron models for the vision template ? (If changed, also change tutorial)

### Template - Numerical


### Template - Computer Vision
