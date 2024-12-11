# B spline
# place knot at xr and every dx years before and after
bspline <- function(x) {
  xl <- min(x) - 10
  xr <- max(x) + 10
  dx <- 2.5
  bdeg <- 3
  knots <- -1*rev(seq(-1*(xr + bdeg * dx), -1*(xl - bdeg * dx), by = dx))
  B <- splines::spline.des(knots, x, bdeg + 1, 0 * x, outer.ok = TRUE)$design
  return(B)
}