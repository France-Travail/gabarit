# TODO list


### Overall

- Add more options with Jinja ? For example --without-tensorflow would remove all references to TensorFlow. Not easy to test.
- Add other templates ? Such as unsupervised templates.
- Fix all TODOs
- Add a note in the README.md about torch packages management (cf. https://stackoverflow.com/questions/57689387/equivalent-for-find-links-in-setup-py)


### All templates

- Review how some models' parameters are managed. For example, the models' learning rate should probably be an __init__ argument.
- Are we checking all parameters the same way ?
- Add ROC curves in plots metrics.
- Add a "model" that would aggregate several models (weak learners) into a meta model.
- We should probably remove nb_iter_keras stuff. It is not used.
- Review how the `get_classes_from_proba` and `inverse_transform` functions work! It is not very understandable!
- Add used preprocessing as models' attributes.
- Remove / fix some `# type: ignore`.
- Rework models folder hierarchy. We should add a directory per library (keras, sklearn, ...), and exploratory subdirectories.
- Some 0_....py scripts are not tested (in functional tests).


### Template - NLP

- Should we keep the `with_new_embedding` argument ? It is not used.
- We should probably mutualise all keras models' `predict_proba` functions in `model_keras.py`. Only `ModelTfidfDense` is different, it should overload this method.
- Rework `model_rules.py`.
- Download flaubert_small_cased before using tests (in Actions)
- Installation error with `python setup.py develop` when requirements.txt is not called first -> error: requests 2.28.0 is installed but requests<2.25.1,>=2.23.0 is required by {'words-n-fun'}


### Template - Numerical


### Template - Computer Vision

- Many unit test to add:
	- `read_folder`
	- `read_folder_object_detection`
	- `rebuild_metadata_object_detection`
	- `rebuild_metadata_classification`
	- Many `utils_object_detector` functions
	- `utils_faster_rcnn` tests
	- etc.
- Implement predict for object detection tasks
