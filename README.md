# UR_handson
```bash
Dockerfile is in ./dockerfiles/Courses/RL/Dockerfile
requirements.txt is in ./projects/RL/requirements.txt


# clone this repo
git clone https://github.com/Bill-Huangz/UR_handson.git

# clone aup-learning-cloud repo, or cd to it. 
git clone https://github.com/AMDResearch/aup-learning-cloud.git
cd aup-learning-cloud

# copy all the files
cp UR_handson/* aup-learning-cloud/ -r

# build images
cd ./aup-learning-cloud/dockerfiles/Courses/RL
./build.sh
cd ../../../ # To the root of auplc

# install locally
./auplc-installer

### In the browser terminal ### 
alias python='python3' 
cd RL
python train.py --alg SAC --cuda

```
