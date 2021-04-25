# _Realtime Artistic Style Transfer: An Attempt in Python_

### (NOTE: Work in progress - both README and code...)

## Project Motivation and Process
- **Neural Style Transfer** is one of the most fascinating, flashy uses of Computer Vision techniques. Taking a photo and turning it into a beautiful watercolor style image, or something echoing "Starry Night", it's attention-grabbing stuff.
- Doesn't hurt that the technology and techniques behind style transfer are rather interesting, with this project implementing **[Magenta](https://arxiv.org/abs/1705.06830)** via TF Hub's Arbitrary Image Stylization module.
- Personally I have been interested in RNNs and DCGANs for this task, but the implementation here works and is impressively performant. RNNs have given me enough joy in generating text anyway _(see my projects on here generating text from Deleuze, Joyce, etc...)_

# Run Instructions:
- Developed using __Python 3.7.3__, on your local environment use pip3 to install from the Requirements.txt.
- Run **app.py** using Python, OpenCV will open a window from your primary webcam's feed and apply the style transfer. _Press Q or Esc to quit the window._

# Future Work
- I would love to get this deployed as an app for it to be easier for people to play with. I see hurdles in the computational overhead, the streaming API in JS, and more. 
- Some [promising work](https://discuss.streamlit.io/t/new-component-streamlit-webrtc-a-new-way-to-deal-with-real-time-media-streams/8669) has been done in Streamlit with realtime web streams, however, I've yet to adapt the code to be performant utilizing this groundwork enough for deployment.
- With an app or even a more developed command-line version would come user options for style images, the option to select a secondary webcam, bunch of stuff I've thought of down the line for this.

# References
- [1] Golnaz Ghiasi, Honglak Lee, Manjunath Kudlur, Vincent Dumoulin, Jonathon Shlens. Exploring the structure of a real-time, arbitrary neural artistic stylization network. Proceedings of the British Machine Vision Conference (BMVC), 2017.
- https://www.pyimagesearch.com/2015/12/21/increasing-webcam-fps-with-python-and-opencv/
- https://towardsdatascience.com/fast-neural-style-transfer-in-5-minutes-with-tensorflow-hub-magenta-110b60431dcc
- https://github.com/lengstrom/fast-style-transfer
- https://medium.com/hackernoon/diy-prisma-fast-style-transfer-app-with-coreml-and-tensorflow-817c3b90dacd
- https://github.com/whitphx/streamlit-webrtc