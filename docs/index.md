# Gabarit - Templates Data Science

Gabarit provides you with a set of python templates (a.k.a. frameworks) for your Data Science projects. It allows you to generate a code base that includes many features to speed up the production and testing of your AI models. You just have to focus on the core of Data Science.

## Philosophy

As a team, we strive to help Data Scientists across the board (and ourselves!) build awesome IA projects by speeding up the development process. This repository contains several frameworks allowing any data scientist, IA enthousiast (or developper of any kind, really) to kickstart an IA project from scratch.  

We hate it when a project is left in the infamous POC shadow valley where nice ideas and clever models are forgotten, thus we tried to pack as much production-ready features as we could in these frameworks.  

As Hadley Wickhman would say: "you can't do data science in a GUI". We are strong believers that during a data science or IA project, you need to be able to fine tune every nooks and crannies to make the best out of your data.  

Therefore, these frameworks act as project templates that you can use to generate a code base from nothing (except for a project name). Doing so would allow your fresh and exciting new project to begin with loads of features on which you wouldn't want to focus this early :
- Built-in models: from the ever useful TF/IDF + SVM to the more recent transformers
- Model-agnostic save/load/reload : perfect to embed your model behind a web service
- Generic training/predict scripts to work with your data as soon as possible
- DVC & MLFlow integration (you have to configure it to point to your own infrastructures)
- Streamlit demo tool
- ... and so much more !

### Frameworks

Gabarit contains the following frameworks :

#### [**NLP**](/frameworks/NLP) to tackle classification use cases on textual data
  -	Relies on the Words'n fun module for the preprocessing requirements
  - Supports :
      - Mono Class / Mono Label classification
      - Multi Classes / Mono Label classification
      - Mono Class / Multi Labels classification

#### [**Numeric**](/frameworks/NUM) to tackle classification and regression use cases on numerical data
  - Supports :
    - Regression
    - Multi Classes / Mono Label classification
    - Mono Class / Multi Labels classification

#### [**Computer Vision**](/frameworks/VISION) to tackle classification use cases on images
  - Supports
    - Mono Class / Mono Label classification
    - Multi Classes / Mono Label classification
    - Area of interest detection

#### [**API**](/frameworks/API) for exposing your model to the world
  - Supports
    - Gabarit model created thanks to one of the previous package
    - A model of your own
  - Provides
    - A [FastAPI](https://fastapi.tiangolo.com/) to expose your model

These frameworks have been developped to manage different subjects but share a common structure and a common philosophy. Once a project made using a framework is in production, any other project can be sent into production following the same process.
Along with these frameworks, an API template has been developped and should soon be open sourced as well. With it, you can expose framework made models in no time !