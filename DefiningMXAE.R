#' Helper function to create a customized metric
#'
#' 
mx.metric.custom <- function(name, feval) {
  init <- function() {
    c(0, 0)
  }
  update <- function(label, pred, state) {
    m <- feval(label, pred)
    state <- c(state[[1]] + 1, state[[2]] + m)
    return(state)
  }
  get <- function(state) {
    list(name=name, value=(state[[2]]/state[[1]]))
  }
  ret <- (list(init=init, update=update, get=get))
  class(ret) <- "mx.metric"
  return(ret)
}
#' MXAE (Max Absolute Error) metric for regression
#'
#'@export
mx.metric.mxae <- mx.metric.custom("mxae", function(label, pred) {
  pred <- mx.nd.reshape(pred, shape = 0)
  res <- mx.nd.max(mx.nd.abs(label-pred))
  return(as.array(res))
})

