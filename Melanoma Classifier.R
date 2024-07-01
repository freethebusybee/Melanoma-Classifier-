library(keras)
library(tensorflow)
library(shiny)
library(EBImage)

# Set paths
melanoma_path <- "/Users/farihasyed/Downloads/Skin Cancer/melanoma"
non_melanoma_path <- "/Users/farihasyed/Downloads/Skin Cancer/non-melanoma"

# Function to load and preprocess images
load_images <- function(path, label) {
  files <- list.files(path, full.names = TRUE)
  images <- lapply(files, function(f) {
    img <- readImage(f)
    img <- resize(img, 224, 224)
    img <- as.array(img)
  })
  images <- array_reshape(do.call(abind, c(images, list(along = 1))), c(length(files), 224, 224, 3))
  list(images = images, labels = rep(label, length(files)))
}

# Load and preprocess data
melanoma_data <- load_images(melanoma_path, 1)
non_melanoma_data <- load_images(non_melanoma_path, 0)

# Combine data
x <- abind(melanoma_data$images, non_melanoma_data$images, along = 1)
y <- c(melanoma_data$labels, non_melanoma_data$labels)

# Split data
indices <- sample(1:nrow(x))
train_indices <- indices[1:floor(0.8 * length(indices))]
test_indices <- indices[(floor(0.8 * length(indices)) + 1):length(indices)]

x_train <- x[train_indices,,,,drop=FALSE]
y_train <- y[train_indices]
x_test <- x[test_indices,,,,drop=FALSE]
y_test <- y[test_indices]

# Create model using TensorFlow directly
input_layer <- tf$keras$layers$Input(shape = list(224L, 224L, 3L))
x <- tf$keras$layers$Conv2D(filters = 32L, kernel_size = list(3L, 3L), activation = "relu")(input_layer)
x <- tf$keras$layers$MaxPooling2D(pool_size = list(2L, 2L))(x)
x <- tf$keras$layers$Conv2D(filters = 64L, kernel_size = list(3L, 3L), activation = "relu")(x)
x <- tf$keras$layers$MaxPooling2D(pool_size = list(2L, 2L))(x)
x <- tf$keras$layers$Conv2D(filters = 64L, kernel_size = list(3L, 3L), activation = "relu")(x)
x <- tf$keras$layers$Flatten()(x)
x <- tf$keras$layers$Dense(units = 64L, activation = "relu")(x)
output <- tf$keras$layers$Dense(units = 1L, activation = "sigmoid")(x)

model <- tf$keras$models$Model(inputs = input_layer, outputs = output)

# Compile model
model$compile(
  optimizer = "adam",
  loss = "binary_crossentropy",
  metrics = list("accuracy")
)

# Convert data to TensorFlow tensors
x_train <- tf$constant(x_train, dtype = tf$float32)
y_train <- tf$constant(y_train, dtype = tf$float32)

# Train model
history <- model$fit(
  x_train, y_train,
  epochs = 10L,
  batch_size = 32L,
  validation_split = 0.2
)

# Define UI
ui <- fluidPage(
  titlePanel("Melanoma Classifier"),
  fileInput("file", "Choose an image file"),
  imageOutput("image"),
  textOutput("result")
)

# Define server logic
server <- function(input, output) {
  output$image <- renderImage({
    req(input$file)
    list(src = input$file$datapath, width = 300)
  }, deleteFile = FALSE)
  
  output$result <- renderText({
    req(input$file)
    img <- readImage(input$file$datapath)
    img <- resize(img, 224, 224)
    img <- array_reshape(img, c(1, 224, 224, 3))
    img <- tf$constant(img, dtype = tf$float32)
    
    pred <- model$predict(img)
    
    if(pred[1] > 0.5) {
      "This image is classified as Melanoma"
    } else {
      "This image is classified as Non-Melanoma"
    }
  })
}

# Run the app
shinyApp(ui, server)
