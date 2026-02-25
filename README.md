# IE4228-Assignment
## Assignment 1
Design and build up a human face recognition system that can detect and recognize a person with a live camera. This is a team-work at lab S2-B5C-02 and each team should consist of 5~8 students. You are encouraged to help each other within the team and cross the teams except for copying each other. Program can be inMATLAB, Python, or C/C++. 

Collect face images of all team members and a few others as gallery images to detect a person’s face lively and recognize it into one or none in the gallery correctly. Over 10 face images for each person should be collected in gallery. Recommend at least 90X90 pixels for each cropped face image in gallery.
You can also download some face database from internet to help you training PCA or LDA or the both.
Convert colour to grey image for recognition. Some face image normalization should be performed as pre-processing (illumination, spatial position, rotation and scale normalization). Other image enhance techniques to improve the face quality are optional.
The trained face detection model such as “Viola & Jones” from OpenCV can be called, providing face location and size. The detected faces will be resized to the same size as you did during the face recognition training process and be recognized by your designed and trained recognition model.
You can optionally find some face and nonface dataset from internet and train your own face detection model such as PCA.
You may opt for other face detection and recognition algorithms except for deep learning model. But you must have full understanding of the proposed algorithm and well elaborate it in the report and group presentation/demonstration.
The assessment will be conducted by oral presentation, demonstration and interview in week 12 & 13 in the Lab. A live demonstration GUI is required. During demonstration, all team member should pose in front of the camera for detection and recognition.
Half of teams of each lab class will do the demo of assignment 1 in week 12 and assignment 2 in week 13 while the other half of teams will do the demo of assignment 2 in week 12 and assignment 1 in week 13.
A detailed technical report per team, which describes your project and performance analysis, source code and a list of individual contributions of each member and signed by each member, must be submitted in one PDF or WORD file per team to NTULearn before the week of your lab assessment.
Each student submits 3-5 minutes video for each Assignment for individual assessment by the end of week 13.

## Assignment 2
Design and implement a deep learning-based human face detection and recognition system that can accurately detect and recognize faces in real-time using a live camera. You will explore and utilize deep learning methods, e.g., FaceNet and MTCNN, while being encouraged to research and experiment with other state-of-the-art deep learning models and techniques.

Collect face images of all team members and a few additional individuals to build a database for live face detection and accurate recognition. Ensure that each person in the database has more than 10 face images.
For the live demo, faces should be highlighted with bounding boxes and correctly identified if they are present in the database. If the face is not in the database, an "unknown" signal should be displayed.
You may directly utilize off-the-shelf face detection models, such as MTCNN, and face recognition models, like FaceNet. However, if you wish to go further, you may opt to fine-tune these models using the face images you have collected for improved performance.
Additionally, you may explore publicly available face detection and recognition datasets, such as VGGFace, to train your own face detection and recognition network.
If you choose to implement other face detection and recognition algorithms based on deep neural network architectures, ensure you have a comprehensive understanding of the chosen methods. Provide a detailed explanation in your report and during the group presentation or demonstration.
The assessment will be conducted by oral presentation, demonstration and interview in week 12 & 13 in the Lab. A live demonstration GUI is required. During the demonstration, all team members should pose in front of the camera for detection and recognition.
Half of teams of each lab class will do the demo of assignment 1 in week 12 and assignment 2 in week 13 while the other half of teams will do the demo of assignment 2 in week 12 and assignment 1 in week 13.
A detailed technical report per team, which describes your project and performance analysis, source code and a list of individual contributions of each member and signed by each member, must be submitted in one PDF or WORD file per team to NTULearn before the week of your lab assessment.
Each student submits 3-5 minutes video for each Assignment for individual assessment by the end of week 13.
