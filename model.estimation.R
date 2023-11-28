require(rebmix)
require(imager)
require(yaml)
require(argparser)
# Create a parser
p <- arg_parser("Round a floating point number")

# Add command line arguments
p <- add_argument(p, "--cwd", help="current working directory", type="character", default=getwd())
p <- add_argument(p, "--images-dir", help="directory with images", type="character", default="images/")
p <- add_argument(p, "--pdf", help="probability density function", type="character", default="normal")
p <- add_argument(p, "--cmax", help="maximum number of components", type="integer", default=64)
p <- add_argument(p, "--cmin", help="minimum number of components", type="integer", default=1)

argv <- parse_args(p)

setwd(argv$cwd)

images.dir <- argv$images_dir

pdf <- argv$pdf
cmax <- argv$cmax
cmin <- argv$cmin

K <- 256
EM <- new("EM.Control", strategy = "single")
IC <- "BIC"

files <- list.files(images.dir)

N <- length(files)

for (i in 1:N){
  
  #print(paste0(dir, files[i]))
  
  data <- load.image(paste0(images.dir, files[i]))
  
  data <- grayscale(data)
  
  data <- data.frame(x=as.double(data)*255)
  
  if (i == 1) {
    Hist <- fhistogram(Dataset = data, K = K, ymin = -0.5, ymax = 255.5, shrink = FALSE)
  }
  else {
    Hist <- fhistogram(x = Hist, Dataset = data, shrink = FALSE)
  } 
}

model <- REBMIX(Dataset = list(Hist),
                    cmax = cmax,
                    Criterion = IC,
                    pdf = pdf,
                    theta3 = NA,
                    EMcontrol = EM)

clustering <- RCLRMIX(x=model)


write.table(clustering@tau, file="D.txt", sep= " ", row.names = F, col.names = F)