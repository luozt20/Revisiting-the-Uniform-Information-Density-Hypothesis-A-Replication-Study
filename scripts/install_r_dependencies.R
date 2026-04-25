args <- commandArgs(trailingOnly = FALSE)
script_arg <- grep("^--file=", args, value = TRUE)
script_path <- normalizePath(sub("^--file=", "", script_arg[[1]]))
repo_root <- normalizePath(file.path(dirname(script_path), ".."))
project_lib <- file.path(repo_root, ".Rlibs")

dir.create(project_lib, recursive = TRUE, showWarnings = FALSE)
.libPaths(c(project_lib, .libPaths()))

packages <- c("lme4", "lmerTest", "ggplot2", "dplyr")
repos <- "https://cloud.r-project.org"

missing <- packages[!vapply(packages, requireNamespace, logical(1), quietly = TRUE)]

if (length(missing) == 0) {
  cat("All required R packages are already installed.\n")
} else {
  install.packages(missing, repos = repos, lib = project_lib)
}
