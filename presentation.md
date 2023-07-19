<!DOCTYPE html>
<html>
<head>
  <!-- Reveal.js dependencies -->
  <link rel="stylesheet" href="https://revealjs.com/dist/reveal.css">
  <script src="https://revealjs.com/dist/reveal.js"></script>

  <!-- Custom CSS -->
  <style>
    /* Autofit text */
    .reveal .slides {
      display: flex;
      align-items: center;
      justify-content: center;
    }

    .reveal .slides section {
      font-size: 2vw; /* Adjust the font size as needed */
    }

    /* Autofit images */
    .reveal .slides img {
      max-width: 100%;
      max-height: 100%;
      object-fit: contain;
    }
  </style>
</head>
<body>
  <div class="reveal">
    <div class="slides">

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
        <h1>Satellite Image Coverage Classification Using ResNet Convolution Neural Network</h1>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes="Introductions">
    <h2>Credits</h2>
    <p>Presented by:</p>
    <div class="headshots">
      <img src="C:/Users/calve/Downloads/static_visuals/emily.jfif" alt="Emily Calvert Headshot">
      <img src="C:/Users/calve/Downloads/static_visuals/sophie.jfif" alt="Sophie Ollivier Salgado Headshot">
    </div>
    <p>Emily Calvert
       Email: calvertemily15@gmail.com
       LinkedIn: <a href="https://www.linkedin.com/in/emily-calvert-data/">Emily Calvert</a></p>
    <p>Sophie Ollivier Salgado
       Email: sollivier5@gmail.com
       LinkedIn: <a href="https://www.linkedin.com/in/sophie-ollivier-salgado-a45552128/">Sophie Ollivier Salgado</a></p>
    <p>Contact us with any questions or to connect!</p>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <p><b>Advantages and Applications of GeoSpatial Analytics and AI using Satellite Imagery</b></p>
    <p>Geospatial analytics combined with Artificial Intelligence (AI) has become a game-changer in many industries. This combination empowers us to derive critical insights and patterns from vast volumes of satellite imagery, which can be incredibly beneficial for a variety of applications. It is essentially reshaping how we understand and interact with our world, helping us solve complex problems by illuminating new insights.</p>
    <div class="illustration">
      <img src="C:/Users/calve/Downloads/static_visuals/sattelite.png" alt="Satellite" width="400">
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h2>General</h2>
    <p>Satellite imagery can cover virtually every corner of the globe, allowing for large-scale analysis. This can lead to rapid insights across vast geographies.</p>
    <ul>
      <li>As satellites pass over the same locations multiple times, they create a historical archive of images. We can analyze these images to identify changes and trends over time, providing powerful insights into patterns of growth, decline, or transformation.</li>
    </ul>
    <img src="path/to/illustration.jpg" alt="Illustration" style="max-width: 70%; margin: 20px auto;">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h2 style="font-size: 2.5em; font-weight: bold;">Examples of Industry Applications</h2>
    <div style="display: flex; justify-content: center;">
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Agriculture</h3>
    <p>By analyzing satellite imagery, farmers can identify areas of stress in crops long before they might be visible to the naked eye. This can lead to early intervention, potentially saving vast swathes of crops from disease or pest infestation.</p>
    <img src="path/to/illustration.png" alt="Agriculture Illustration" class="illustration">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Environmental Science</h3>
    <div class="illustration">
      <!-- Insert your engaging illustration here -->
      <!-- Add appropriate image HTML code or embed external content -->
    </div>
    <p>This technology is being used to monitor deforestation, track wildlife populations, and assess the impact of natural disasters. We can analyze satellite data to help direct emergency services to the most affected areas.</p>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Urban Planning and Infrastructure</h3>
    <p>Planners can analyze satellite images and gain insights into population growth, land use changes, and infrastructure development. This can help inform decisions about where to build new roads, schools, and other public infrastructure.</p>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h2 class="slide-title">Conclusion</h2>
    <div class="illustration">
      <!-- Insert your engaging illustration here -->
      <img src="path_to_your_image.png" alt="Illustration">
    </div>
    <p>Geospatial analytics and AI using satellite imagery offer powerful tools for gaining insights and solving complex problems across a wide range of industries. These technologies allow us to understand our world in greater detail and make more informed decisions about how to manage our resources and plan for the future.</p>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1>Problem Statement</h1>
    <p>Manual Classification of Satellite Imagery is both expensive and time-consuming, yet it is a vital task for identifying cloud coverage and determining the quality of satellite data.</p>
    <!-- Insert your illustration here -->
    <div class="illustration">
      <!-- Place your illustration code or image here -->
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h2>The Importance of Classification</h2>
    <p>Without proper classification, we are left with single snapshots of imagery for our analysis. This leads to unrepeatable results and highly localized analyses.</p>
    <div class="illustration">
      <!-- Place your illustration code or image here -->
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Challenges in Manual Classification</h3>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Volume of Data</h3>
    <p>Satellite imagery is incredibly dense and memory intensive. With 50 years of accessible satellite data, and more being collected every hour, the potential scale for geospatial analysis is vast. However, computing and processing power presents a significant constraint.</p>
    <img src="path_to_your_illustration_image.jpg" alt="Volume of Data Illustration">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Time Consumption</h3>
    <p>Given the scale and pixel density, manually identifying each raster in an image can take hours.</p>
    <!-- Place your illustration here -->
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h3>Other obstacles and considerations include:</h3>
    <ul>
      <li>Consistency: Human perception can lead to inconsistent classification methodologies employed, leading to unreproducible results.</li>
      <li>Cost: Given the manpower and computing power necessary, manual preprocessing of satellite imagery will cost your organization greatly.</li>
    </ul>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1>Our Solution</h1>
    <p>We propose a model for the classification of coverage type in satellite imagery. This automated pre-processing tool will determine the quality of imagery by identifying cloud coverage and other relevant features, depending on your organization’s needs.</p>
    <img src="path/to/image.jpg" alt="Engaging Image" style="width: 60%; margin: auto;">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "Our project addresses the above concerns by producing a model for classification of coverage type in satellite imagery. This automation of preprocessing will determine quality of imagery by identifying cloud coverage. Then depending on your organization’s needs we can identify vegetation, desert, or water coverage. Applications are endless. Below are some examples on how automating classification of imagery will contribute to your organization.">
    <h2>Applications</h2>
    <p>The possibilities for applying this technology are endless. This slide highlights some examples of how automated classification of imagery will contribute to your organization.</p>
    <div style="text-align: center;">
      ![Engaging Image](image.jpg)
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "With the proper data management, tuning of hyperparameters, and data augmentation, this mode has the potential to meet the needs of the project at hand. Deploying the model for classification will give your organization the ability to process more data and build more robust models. By replacing manual classification and tokenizing of raster layers with automated processes, your organization can use more detailed imagery and larger datasets.">
    <h3>Scalability</h3>
    <p>Our model is designed to meet the needs of any project. With the right data management, tuning of hyperparameters, and data augmentation, you can process more data and build more robust models. Our solution enables the use of more detailed imagery and larger datasets.</p>
    <img src="path/to/your/image.jpg" alt="Engaging Image" style="max-width: 80%; margin-top: 1rem;">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "Our model can be deployed and evaluate the quality of imagery and pertinence to your project prior to moving dense data across platforms, making large api calls/orders, and preprocessing whole sets of imagery.">
    <h3>Efficiency</h3>
    <p>The model can quickly evaluate the quality and relevance of imagery before transferring dense data or making large API calls. This process increases efficiency by focusing resources only on pertinent data sets.</p>
    <img src="path/to/your/image.jpg" alt="Engaging Image" style="max-width: 60%; margin: 30px auto;">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "By identifying an appropriate model using distribution patterns we can produce a model that will follow the same processes of classification with each iteration. This leads to reproducible results.">
    <h3>Consistency</h3>
    <p>Our model ensures consistency by employing the same processes of classification with each iteration. This leads to reproducible results.</p>
    <img src="path/to/image.jpg" alt="Engaging Image" style="width: 50%; margin-top: 2rem;">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "All of these innovations lead to money and time saved. In addition, it enables applications in industries such as disaster management and emergency response, intelligence and national security,  traffic management, logistics and freight management, and wildfire surveillance. These are industries where near real time analysis is necessary, which is impossible with manual techniques.">
    <h3>Conclusion</h3>
    <p>Our innovative solution leads to both time and cost savings. Additionally, it opens up possibilities for real-time analysis in various industries, such as disaster management, national security, traffic management, logistics, and wildfire surveillance.</p>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1>Model</h1>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1>Data Selection</h1>
    <ul>
      <li>Source: [Kaggle](https://www.kaggle.com/datasets/mahmoudreda55/satellite-image-classification)</li>
      <li>Data was manually sorted for training by the dataset author</li>
      <li>Data was gathered from various sensors and Google Maps</li>
      <li>The dataset comprises 5631 images in jpg format</li>
    </ul>
    <img src="C:/Users/calve/Downloads/static_visuals/data_source.jpg" alt="Kaggle Header" style="max-width: 70%; margin-top: 2em;">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "We have 24.6% more data on the classes that aren’t desert. This might negatively affect the model by producing a bias against the minority class or poor detection of anomalies due to overfitting the minority class/insufficient data to establish a distribution pattern for that class. There’s not a significant imbalance so we will run the model without addressing the class imbalance. After evaluating the performance metrics we might come back and try different techniques to adjust for the imbalance. We started with a standard split that can of course be revisited based on model performance. The code we wrote makes 4 new directories. Source (original data), Training: 60%, For our predictions to be accurate the model should be trained on the majority of the data we have available. Validation: 20% We have 20% of the data as a blind set to allow us to finetune the hyperparameters if performance is subpar. Test: 20% The last 20% will provide the final performance metrics. The data size also suggest a descrepancy in sample detail, thus contributing to imbalance">
    <h1>Class Imbalance</h1>
    <ul>
      <li>We observed a class imbalance with 24.6% more data for classes other than 'desert'</li>
      <li>We will initially run the model without addressing this imbalance, but may revisit this decision based on performance metrics</li>
      <li>The data split was as follows: Training (60%), Validation (20%), Test (20%)</li>
    </ul>
    <div class="image-container">
      <img src=""C:/Users/calve/Downloads/static_visuals/class_imbalance_plot.png"" alt="Samples and Classes">
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes="Transform
Transforming the Data is key to making sure the model functions with our data. We used tools in the PyTorch package. Training: Crop 224x224 pixels, function includes scale and ratio augmentation for data integrity. 224 is chosen because it is a standard input size for many pre-trained models including ResNet. Random Horizontal Flip This flips our images to augment the data to help prevent overfitting and make generalizations. This occurs at the end of each epoch the model will decide to randomly flip the images on the horizontal edge. Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  In image processing for machine learning, we often normalize pixel values to help the model learn more effectively. The Normalize transformation in PyTorch adjusts pixel values in each color channel (red, green, and blue) to be centered around 0 and within a standard range. The specific values we use for this, such as [0.485, 0.456, 0.406] for means and [0.229, 0.224, 0.225] for standard deviations, are calculated from the ImageNet dataset, which is commonly used to train these models. By using these same values, we ensure our image processing matches the conditions under which the model was originally trained, helping it to perform better on our data. Test and Validation Resize 256x256 pixels  Resizing all images to the same size is necessary because input images may have different sizes. Our model can handle images of differing sizes, but to reduce processing power we resized everything uniformly.
Center Crop 224x224 pixels test it in a standard and consistent way (center crops)">
    <h1>Transforming the Data</h1>
    <ul>
      <li>Crop to 224x224 pixels for training data, with scale and ratio augmentation for data integrity</li>
      <li>Random horizontal flip to augment the data and prevent overfitting</li>
      <li>Normalization of pixel values using pre-calculated values from the ImageNet dataset</li>
    </ul>
    <img src=""C:/Users/calve/Downloads/static_visuals/data_transform.jpg"" class="top-right-image" />
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes="Amount and Type of Data  A large amount of data can indicate necessity for a deeper neural network that can learn more complex patterns. The type of data also matters: is it tabular, image, text, or some other type of data? For image data, convolutional neural networks (CNNs). Distribution Pattern Pixel Intensity Distribution Desert The pixel intensity distribution for desert images shows a high mean value of 0.608, indicating that desert scenes are generally brightly lit. The range of pixel intensities, from a minimum of 0.418 to a maximum of 0.722, is relatively small. With a standard deviation of 0.058, the intensities appear to be closely grouped around the mean. WaterWater scenes exhibit a low mean pixel intensity of 0.272, possibly due to deeper water absorbing more light. The standard deviation is higher at 0.085, indicating a wider spread of pixel intensities. The pixel intensities range from 0.166 to 0.519, indicating a diversity in lighting conditions for water scenes. Green Area
  For green area images, the mean pixel intensity is even lower than for water, at 0.235. This could be due to the light absorption properties of plants. The standard deviation is very low at 0.025, suggesting that pixel intensities are very consistent within green areas. This is likely due to the consistent color of vegetation. The range of pixel intensities, from 0.169 to 0.361, is the narrowest of all categories. Cloudy Cloudy scenes present the highest mean pixel intensity of all categories at 0.613. This could be due to light scattering from the clouds. However, the standard deviation is also the highest at 0.118, indicating a wider range of pixel intensities. This variation might be due to differing levels of cloud cover and thickness. The range of pixel intensities is also the widest, from 0.089 to 0.959, suggesting a large diversity in lighting conditions for cloudy scenes.Color Channel Distributions In terms of color channel distributions, different patterns emerge for each category.Desert The mean values for R, G, and B color channels in desert scenes are 0.500, 0.608, and 0.718 respectively. The higher mean values in G and B channels may reflect the unique color properties of desert landscapes. Water For water scenes, the mean values are 0.347, 0.284, and 0.186 for the R, G, and B channels respectively. This could reflect the deeper and darker tones generally found in water bodies. Green Area Green areas show mean values of 0.298, 0.253, and 0.154 for R, G, and B color channels respectively. The low mean values, especially in the B channel, may be due to the dominance of green vegetation in these areas. Cloudy Cloudy scenes show the mean values for R, G, and B color channels to be 0.587, 0.628, and 0.624 respectively. This distribution, with fairly similar values across all three channels, could be due to the diffuse lighting conditions generally found in cloudy environments. This analysis provides a descriptive overview of the pixel intensities and color channel distributions for each of the four categories. These insights could be useful for further data analysis or to inform image processing techniques. Conclusion Distribution Pattern & Pixel Density: The distribution pattern of pixel intensity can help us understand the characteristics of different classes in our dataset. For example, a class like 'cloudy' has a wide range of pixel intensities and higher standard deviation, indicating more variation within this class. This could mean that a model will need to be more robust and flexible to handle such variability. On the other hand, classes like 'desert' and 'green' have lower standard deviations, indicating less intra-class variation, which could be easier for the model to learn. Moreover, understanding the average pixel density can help us in setting appropriate thresholds for image segmentation and further processing.
  Understanding the color channel distributions is vital, particularly in RGB images where different color channels can provide distinct information. If a certain class has higher intensities in a particular color channel consistently (like 'blue' for 'desert'), the model could potentially leverage this information to improve accuracy. For example, if a model is poor at using color information, you might opt for a different model that can better handle this aspect of the data. Goal The goal of a model is a key factor that guided the model selection process. The type of problem we are trying to solve, the metrics we aim to optimize, and the context in which the model will be used heavily influenced our decision. Since the goal of the mode is to accurately classify different types of geographical features in satellite imagery (like 'water', 'desert', 'cloudy', 'green'), then we want a model that can perform this multi-class classification effectively. Given that we're dealing with images, convolutional neural network (CNN) models are often a good choice due to their ability to capture spatial relationships and recognize patterns in image data.">
    <h1>Factors Influencing Model Selection</h1>
    <ul>
      <li>Amount and Type of Data: A large amount of image data demands a model capable of learning complex patterns.</li>
      <li>Distribution Pattern: Pixel intensity distribution varies between classes; some classes have a wide range of pixel intensities, indicating more variability.</li>
      <li>Pixel Density and Color Channel Distributions: Understanding these can help set appropriate thresholds for image segmentation and processing, and can indicate which models might be more effective.</li>
      <li>Goal: The primary goal is accurate multi-class classification of satellite images.</li>
    </ul>
    <figure>
      <img src=""C:\Users\calve\Downloads\static_visuals\pixel_color_distribution.png" alt="Grid of Distribution Plots">
    </figure>
  </section>

---

  <!-- Insert Interactive Visual Here  -->

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes="How it works
ResNet, short for Residual Network, is a type of Convolutional Neural Network (CNN) for image recognition. As we add more layers to a CNN, the network becomes more capable of learning complex patterns in the data.  When a network gets too deep, it suffers from something called the vanishing gradient problem. This is like trying to hear a whisper in a noisy room - the important details (gradients) become too small to detect and learn from, making the network hard to improve. ResNet tackles the vanishing gradient problem by using a skip connection.Let's imagine a set of layers in the neural network as a box that does some complex transformations on the data we feed into it. The data, or the 'input', is like a ball that we throw into this box. After the ball has bounced around inside and undergone changes (the transformations), it comes out the other side - the 'output'.Now, in a regular neural network, each layer would just take the output from the previous layer and feed it into its own set of transformations. The ball would go into one box, come out transformed, and immediately go into the next box. In a ResNet, however, we have an additional path - the 'skip connection' or 'shortcut'. Alongside the usual path (ball goes into box, gets transformed, comes out), we have a straight path that bypasses the box. It's like throwing two balls - one goes into the box, the other travels straight to the end. At the end, we combine both balls - the one that went through the box and the one that bypassed it. This combination is the final output that is fed into the next layer. The effect of this setup is that even if the transformation in the box is very complex and the ball that goes through it comes out very changed, we always have the original ball that bypassed the box. This way, we keep some of the original, unaltered information which can help the network learn more effectively, especially when it is very deep. This is the residual part of the ResNet - we add the original input (the residue) to the output. This seemingly simple idea has a profound impact. The core idea behind ResNet is essentially that deeper networks should perform at least as well as shallower ones. By using these shortcuts, layers are able to learn identity functions that ensure their outputs are at least as informative as their inputs. This means that adding more layers shouldn't hurt performance, and indeed, ResNets can successfully train networks with hundreds (or even thousands) of layers.">
    <h1>ResNet Convolution Neural Network</h1>
    <ul>
      <li>ResNet, short for Residual Network, is a type of Convolutional Neural Network (CNN) for image recognition.</li>
      <li>ResNet uses skip connections to tackle the "vanishing gradient problem" in deep neural networks.</li>
      <li>ResNet can handle a large amount of high-dimensional data, such as images, efficiently.</li>
    </ul>
    <!-- Insert illustration here -->
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "How it addresses above factors. Amount and Type of Data: The dataset involves images, which are a high-dimensional form of data. The ResNet architecture excels with this kind of data due to its design principles - it uses convolutional layers, which are particularly well-suited to image data because they can detect local patterns and capture spatial relationships. Furthermore, with a large dataset, we have enough information to effectively train the deeper structure of a ResNet, allowing the model to learn more intricate patterns and avoid overfitting. Distribution Pattern: From the Exploratory Data Analysis (EDA), we see that pixel intensities and color distributions vary across classes (like 'cloudy', 'desert', 'green', 'water'). Some classes have a wider range of pixel intensities and color distributions (like 'cloudy'), suggesting more variability, while others are more consistent (like 'green'). ResNet, due to its deep structure and the inclusion of skip connections, can handle such variability efficiently. It can learn complex mappings for classes with high variability while its skip connections allow it to smoothly handle classes with less variability. Pixel Density and Color Channel Distributions: Understanding these factors can help in configuring the model. For instance, color channel distributions inform us that different classes have varying intensities in different channels. ResNet's convolutional layers can detect these subtle differences due to their ability to apply filters to different color channels and discern patterns within them. Goal: The ultimate goal here is accurate multi-class classification of satellite images. Given its success in image classification tasks, ResNet's ability to capture intricate patterns in the data and its robustness to overfitting make it a suitable choice.">
    <h2>How ResNet Addresses Key Factors</h2>
    <ul>
      <li>ResNet uses convolutional layers, which are effective with image data, and can handle a large amount of data without overfitting.</li>
      <li>ResNet can handle variability in pixel intensities and color distributions due to its deep structure and skip connections.</li>
      <li>ResNet's convolutional layers can detect subtle differences in color channel distributions, helping to distinguish between different classes.</li>
    </ul>
    <div class="image-container">
      <img src="path/to/your/image.jpg" alt="Illustration" width="600" height="400">
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "Model Building Decision Processes Training and Fine Tuning Each epoch took a really long time but since we were getting well performing metrics early on we added an early stop condition. Other than that the ResNet performed exceptionally and no further tuning was necessary.">
    <div class="top-images">
      <img src="C:/Users/calve/Downloads/static_visuals/early_stop.jpg"" alt="early_stop" class="large-image">
      <h1>Model Building Decision Processes</h1>
      <img src="C:/Users/calve/Downloads/static_visuals/epoch.jpg" alt="epochs" class="large-image">
    </div>
    <ul>
      <li>The model was trained with an early stopping condition to improve training efficiency.</li>
      <li>The ResNet model performed exceptionally well, negating the need for further fine-tuning.</li>
    </ul>
    <div class="bottom-images">
      <img src="C:/Users/calve/Downloads/static_visuals/validate.jpg" alt="validation" class="small-image">
      <img src="C:/Users/calve/Downloads/static_visuals/testing.jpg"" alt="test" class="small-image">
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1>Performance</h1>
    <hr>
    <img src=""C:/Users/calve/Downloads/static_visuals/preformance.jpg" alt="Preformance">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1>Confusion Matrix</h1>
    <ul>
      <li>The diagonal elements represent the instances where the predicted label is equal to the true label, i.e., correct predictions. Off-diagonal elements are those mislabeled by the classifier.</li>
      <li>'Cloudy' and 'Desert' scenes are perfectly classified with no mislabels.</li>
      <li>'Green Area' has some misclassifications, with 49 instances incorrectly predicted as 'Water'.</li>
      <li>'Water' scenes also have minor misclassifications, with one instance each mislabeled as 'Cloudy' and 'Green Area'.</li>
      <li>Overall, the model shows a high degree of accuracy with the majority of instances correctly classified for each category.</li>
    </ul>
    <img src=""C:/Users/calve/Downloads/static_visuals/confusion_matrix_heatmap.png" alt="Confusion Matrix Heatmap">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes="Classification Report: The classification report presents key metrics to evaluate the performance of the classification model. Precision: This indicates the ability of the classifier not to label an instance positive that is actually negative. The model shows high precision scores for all classes, indicating a low rate of false positives.Recall: This indicates the ability of the classifier to find all positive instances. The model shows high recall scores for 'Cloudy' and 'Desert', indicating a low rate of false negatives. However, 'Green Area' and 'Water' have lower recall values, reflecting some misclassifications. F1-Score: This is a weighted harmonic mean of precision and recall. The model's F1 scores are close to 1 for 'Cloudy' and 'Desert', indicating a good balance between precision and recall. However, 'Green Area' and 'Water' show lower F1 scores due to the recall's influence. Support: This is the number of actual occurrences of the class in the dataset. The support scores show a balanced distribution of instances for 'Cloudy', 'Green Area', and 'Water'. 'Desert' has a lower number of instances in the test set. Accuracy: This is the ratio of the total number of correct predictions to the total number of input samples. The model has a high accuracy score of 0.91, indicating it made correct predictions most of the time.">
    <h2>Classification Report</h2>
    <ul>
      <li>Precision: High precision scores for all classes.</li>
      <li>Recall: High for 'Cloudy' and 'Desert', lower for 'Green Area' and 'Water'.</li>
      <li>F1-Score: Close to 1 for 'Cloudy' and 'Desert', lower for 'Green Area' and 'Water'.</li>
      <li>Support: Balanced for 'Cloudy', 'Green Area', and 'Water'. Lower for 'Desert'.</li>
      <li>Accuracy: High score of 0.91.</li>
    </ul>
    <div class="grid-container">
      <div class="grid-item">
        <img src="C:/Users/calve/Downloads/static_visuals/precision.png" alt="Precision">
      </div>
      <div class="grid-item">
        <img src="C:/Users/calve/Downloads/static_visuals/recall.png" alt="Recall">
      </div>
      <div class="grid-item">
        <img src="C:/Users/calve/Downloads/static_visuals/f1.png" alt="F1">
      </div>
      <div class="grid-item">
        <img src="C:/Users/calve/Downloads/static_visuals/support.png" alt="Support">
      </div>
    </div>
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "Conclusion
  The distribution patterns of the pixel intensities and color channels in the images significantly mpacted the performance of the model in the following ways: Variation in Pixel Intensities and Color Channels: The model performance is linked with how pixel intensities and color channels are distributed within each class. For instance, 'Desert' and 'Cloudy' scenes had relatively high mean pixel intensities, indicating brighter images, and the model performed well with these classes. On the other hand, 'Green Area' and 'Water' scenes had lower mean pixel intensities, indicating darker images, and the model had more misclassifications with these classes. It suggests that the model might be better trained to recognize brighter images. Spread of Pixel Intensities and Color Channels: The standard deviations for pixel intensities and color channels also play a role in model performance. 'Desert' and 'Cloudy' scenes had lower standard deviations, indicating less variation in their pixel intensity and color channel values. This lack of variation might have made it easier for the model to classify these images. In contrast, 'Green Area' and 'Water' scenes had higher standard deviations, indicating more variation in their pixel intensity and color channel values. This greater variation might have made these images harder for the model to classify accurately. Range of Pixel Intensities: The range of pixel intensities (i.e., the difference between the maximum and minimum values) could also impact model performance. Classes with narrower ranges ('Green Area' and 'Desert') might be easier to distinguish due to their more consistent lighting conditions. In contrast, classes with wider ranges ('Water' and 'Cloudy') present a greater variety of lighting conditions, which might make them harder for the model to classify. Imbalance in Class Instances: The number of instances in each class (class distribution) can also influence model performance. The model may perform better on classes with more instances as it has more data to learn from. In the given dataset, 'Desert' had fewer instances than the other classes, which could potentially impact the model's learning and subsequent prediction capability for this class.">
    <h1>Conclusion</h1>
    <ul>
      <li>The distribution patterns of pixel intensities and color channels impact model performance.</li>
      <li>Model better trained to recognize brighter images ('Desert' and 'Cloudy').</li>
      <li>Less variation in 'Desert' and 'Cloudy' images assists classification.</li>
      <li>More variation in 'Green Area' and 'Water' images presents challenges for classification.</li>
      <li>Classes with narrower ranges of pixel intensities might be easier to distinguish.</li>
      <li>Imbalance in class instances can influence model performance.</li>
    </ul>
    <img src="path/to/unique_visual.png" alt="Unique Visual">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)" data-notes= "If we were to prepare this model for deployment, we would adjust less iterations for the early stop. We had very good numbers early on. It would also be worth fine tuning to achieve higher performance. We might adjust for the class imbalance, provide synthesized or more data and find a stronger distribution pattern on the lower performing classes.">
    <h1>Next Steps</h1>
    <ul>
      <li>Adjust for early stop with fewer iterations.</li>
      <li>Fine-tune for higher performance.</li>
      <li>Address class imbalance.</li>
      <li>Synthesize or provide more data.</li>
      <li>Identify stronger distribution pattern on lower-performing classes.</li>
    </ul>
    <img src="path/to/your-unique-visual.jpg" alt="Unique Visual">
  </section>

---

  <section data-background="linear-gradient(to bottom right, #002366, #87CEEB)">
    <h1 style="font-size: 72px; font-weight: bold; text-transform: uppercase;">Thank You!</h1>
    <p style="font-size: 24px; font-weight: bold; color: #FFD700;">We appreciate your attention</p>
    <img src="path_to_image" alt="Thank You Image" style="width: 400px;">
  </section>

---

</div>
  </div>
  <script>
    // Initialize the Reveal.js presentation
    Reveal.initialize();
  </script>
</body>
</html>
