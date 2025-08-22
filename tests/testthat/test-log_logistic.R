
library(tidyverse)
library(TMB)
library(childSurv)

options(dplyr.summarise.inform = FALSE)
logit <- function(x) return(log(x/(1 - x)))
expit <- function(x) return(exp(x)/(1 + exp(x)))

# get UN-IGME data and results
unigme_all <- read.csv(testthat::test_path("data/UNIGME-2024.csv"))

test_that("log_logistic runs with no error when VR only", {
  
  # get count data shared by UN-IGME
  data_all <- read.csv(testthat::test_path("data/MasterMDB_20231207.csv"))
  data_nn_all <- read.csv(testthat::test_path("data/dth0yrDBwide_20231207.csv"))
  
  # helper function for summing across sexes, return M+F if unknown is NA, but if all are missing return NA
  sum_non_na <- function(x) {
    if (all(is.na(x))) {
      return(NA)
    } else {
      return(sum(x, na.rm = TRUE))
    }
  }
  
  country <- "Japan"
  start_year <- 1950
  end_year <- 2025
  n_years <- length(start_year:end_year)
  
  data <- data_all %>%
    filter(whoname == country & year >= start_year & year <= end_year) %>%
    mutate(x2 = ifelse(is.na(x2), 0, x2),
           x3 = ifelse(is.na(x3), 0, x3),
           x4 = ifelse(is.na(x4), 0, x4)) %>%
    mutate(x1to4 = x1 + x2 + x3 + x4) %>%
    select(iso3, whoname, year, sex, var, x0, x1, x2, x3, x4, x1to4) %>%
    pivot_wider(id_cols = c(iso3, whoname, year, sex), names_from = var, values_from = starts_with("x")) %>%
    group_by(iso3, whoname, year) %>%
    summarise(across(starts_with("x"), sum_non_na)) %>% # sum across sex, ignore NA from unknown sex cases
    ungroup() %>%
    mutate(use1to4 = T)
  
  # exclude 2011 for Japan per UN IGME
  data <- data %>% filter(year != 2011)
  
  # prep neonatal data
  data_nn <- data_nn_all %>%
    filter(whoname == country & sex == "b" & year >= start_year & year <= end_year) %>%
    mutate(xnn_dth = d0 + d1 + d7) %>%
    filter(!(xnn_dth == 0 & lb > 10000)) %>% # Example: Italy 2021
    select(iso3, whoname, year, xnn_dth, births = lb)
  
  data <- left_join(data, data_nn, by = c("iso3", "whoname", "year")) %>%
    mutate(xpnn_dth = x0_dth - xnn_dth,
           xpnn_pop = x0_pop) # for post-neonatal we'll use 1P0 to get first term of the mean
  
  if (any(data$xpnn_dth < 0, na.rm=T)) {
    issue_years <- data[data$xpnn_dth < 0, ]$year
    warning("Removed ", paste(issue_years, collapse = ", "), " because age 0 deaths < neonatal deaths.")
    data <- data %>% filter(!(year %in% issue_years))
  }
  
  # if we have neonatal and <1, use neonatal and post-neonatal, otherwise use <1
  data$usepnn <- as.logical(!is.na(data$xnn_dth) & !is.na(data$births) & !is.na(data$xpnn_dth))
  data <- data %>%
    mutate(x0_dth = ifelse(usepnn, NA, x0_dth),
           xpnn_dth = ifelse(!usepnn, NA, xpnn_dth))
  
  # if we are using 1to4, drop 1, 2, 3, 4, or else drop 1to4
  data <- data %>%
    mutate_at(vars(x1_dth, x2_dth, x3_dth, x4_dth), ~if_else(use1to4, NA, .)) %>%
    mutate(x1to4_dth = ifelse(!use1to4, NA, x1to4_dth))
  
  data <- data %>%
    pivot_longer(cols = starts_with("x")) %>%
    mutate(name = gsub("x", "", name)) %>%
    separate(name, into = c("age", "var"), sep = "_") %>%
    pivot_wider(id_cols = c(iso3, whoname, year, age, births), names_from = var) %>%
    rename(deaths = dth) %>%
    filter(!is.na(deaths) &
             !(is.na(pop) & age != "nn") &
             !(is.na(births) & age == "nn")) %>%
    mutate(x = ifelse(age == "1to4", 12,
                      ifelse(age == "nn", 0,
                             ifelse(age == "pnn", 1,
                                    12 * suppressWarnings(as.numeric(age))))),
           n = ifelse(age == "1to4", 48,
                      ifelse(age == "nn", 1,
                             ifelse(age == "pnn", 11, 12))),
           deaths = as.integer(round(deaths)),
           pop = as.integer(round(pop)))
  
  data_vr <- list(
    n_obs_vr = nrow(data), # number of unique observations
    time_id_vr = match(floor(data$year), start_year:end_year), # year ID for each observation
    months_vr = data$x, # beginning of age interval in months for each observation
    obs_vr = data$deaths, # deaths for each observation
    n_vr = data$n, # length of age interval in months for each observation
    births = data$births, # births in the year for each observation
    pop = data$pop # population for each non-neonatal age group, else NA
  )
  
  expect_no_error(log_logistic(
    data_vr = data_vr,
    time_model = "rw2",
    count_model = "poisson",
    start_year = 1950,
    end_year = 2025,
    include_iid_in_pred = FALSE
  ))
})

test_that("log_logistic runs with no error when FBH only", {
  load(testthat::test_path("data/fbh_test_data.RData"))
  expect_no_error(log_logistic(
    data_fbh = data_fbh,
    data_pp = data_pp,
    time_model = "rw2",
    start_year = 1954, # Based on Liberia, matches time_id in data
    end_year = 2025
  ))
})

test_that("log_logistic runs with no error with pre-processed only", {
  
  country <- "Syrian Arab Republic"
  data <- unigme_all %>%
    filter(Geographic.area == country &
             Observation.Status == "Included in IGME") %>%
    mutate(STD_ERR = ifelse(is.na(STD_ERR), 0.1*OBS_VALUE, STD_ERR),
           OBS_VALUE = OBS_VALUE / 1000,
           STD_ERR = STD_ERR / 1000)
  
  start_year <- floor(min(data$REF_DATE))
  end_year <- 2025
  
  # data %>% ggplot(aes(x = floor(REF_DATE),
  #                     y = OBS_VALUE,
  #                     color = Series.Name)) +
  #   geom_point() +
  #   geom_errorbar(aes(ymin = OBS_VALUE - 1.96*STD_ERR,
  #                     ymax = OBS_VALUE + 1.96*STD_ERR),
  #                 width = 0) +
  #   theme_bw() +
  #   facet_grid(Indicator~.) +
  #   theme(legend.position = "bottom") +
  #   guides(color = guide_legend(nrow = 12)) +
  #   labs(x = "Year", y = "Mortality rate", color = "")
  
  # logit transformation
  data_trans <- data %>%
    mutate(STD_ERR = (1/(OBS_VALUE - OBS_VALUE^2))*STD_ERR) %>% # delta method
    mutate(OBS_VALUE = logit(OBS_VALUE))
  
  data_pp <- list(
    n_obs_pp = nrow(data),
    time_id_pp = match(floor(data$REF_DATE), start_year:end_year), 
    months_pp = case_when(
      data$Indicator == "Under-five mortality rate" ~ 60,
      data$Indicator == "Infant mortality rate" ~ 12,
      data$Indicator == "Neonatal mortality rate" ~ 1
    ),
    obs_pp = data_trans$OBS_VALUE,
    se_pp = data_trans$STD_ERR
  )
  
  pred <- expect_no_error(log_logistic(
    data_pp = data_pp,
    time_model = "rw2",
    start_year = start_year,
    end_year = end_year
  ))
  
  pred %>%
    ggplot(aes(x = year)) +
    geom_ribbon(data = unigme_all %>%
                  filter(Indicator == "Under-five mortality rate" &
                           Geographic.area == country &
                           Series.Name == "UN IGME estimate"),
                aes(x = REF_DATE, ymin = LOWER_BOUND, ymax = UPPER_BOUND),
                fill = "#984EA3", alpha = 0.3) +
    geom_line(aes(y = u5mr_smoothed_med * 1000)) +
    geom_ribbon(aes(ymin = u5mr_smoothed_lower * 1000,
                    ymax = u5mr_smoothed_upper * 1000), alpha = 0.3) +
    geom_point(data = data %>% filter(Indicator == "Under-five mortality rate"),
               aes(x = floor(REF_DATE), y = OBS_VALUE * 1000, color = Series.Name)) +
    geom_errorbar(data = data %>% filter(Indicator == "Under-five mortality rate"),
                  aes(x = floor(REF_DATE), color = Series.Name,
                      ymin = 1000 * (OBS_VALUE - 1.96*STD_ERR),
                      ymax = 1000 * (OBS_VALUE + 1.96*STD_ERR)), width = 0) +
    theme_bw() +
    labs(x = "Year", y = "U5MR", color = "", title = country) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(nrow = 6))
  
  pred %>%
    ggplot(aes(x = year)) +
    geom_ribbon(data = unigme_all %>%
                  filter(Indicator == "Infant mortality rate" &
                           Geographic.area == country &
                           Series.Name == "UN IGME estimate"),
                aes(x = REF_DATE, ymin = LOWER_BOUND, ymax = UPPER_BOUND),
                fill = "#984EA3", alpha = 0.3) +
    geom_line(aes(y = imr_smoothed_med * 1000)) +
    geom_ribbon(aes(ymin = imr_smoothed_lower * 1000,
                    ymax = imr_smoothed_upper * 1000), alpha = 0.3) +
    geom_point(data = data %>% filter(Indicator == "Infant mortality rate"),
               aes(x = floor(REF_DATE), y = OBS_VALUE * 1000, color = Series.Name)) +
    geom_errorbar(data = data %>% filter(Indicator == "Infant mortality rate"),
                  aes(x = floor(REF_DATE), color = Series.Name,
                      ymin = 1000 * (OBS_VALUE - 1.96*STD_ERR),
                      ymax = 1000 * (OBS_VALUE + 1.96*STD_ERR)), width = 0) +
    theme_bw() +
    labs(x = "Year", y = "IMR", color = "", title = country) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(nrow = 6))
  
  pred %>%
    ggplot(aes(x = year)) +
    geom_ribbon(data = unigme_all %>%
                  filter(Indicator == "Neonatal mortality rate" &
                           Geographic.area == country &
                           Series.Name == "UN IGME estimate"),
                aes(x = REF_DATE, ymin = LOWER_BOUND, ymax = UPPER_BOUND),
                fill = "#984EA3", alpha = 0.3) +
    geom_line(aes(y = nmr_smoothed_med * 1000)) +
    geom_ribbon(aes(ymin = nmr_smoothed_lower * 1000,
                    ymax = nmr_smoothed_upper * 1000), alpha = 0.3) +
    geom_point(data = data %>% filter(Indicator == "Neonatal mortality rate"),
               aes(x = floor(REF_DATE), y = OBS_VALUE * 1000, color = Series.Name)) +
    geom_errorbar(data = data %>% filter(Indicator == "Neonatal mortality rate"),
                  aes(x = floor(REF_DATE), color = Series.Name,
                      ymin = 1000 * (OBS_VALUE - 1.96*STD_ERR),
                      ymax = 1000 * (OBS_VALUE + 1.96*STD_ERR)), width = 0) +
    theme_bw() +
    labs(x = "Year", y = "NMR", color = "", title = country) +
    theme(legend.position = "bottom") +
    guides(color = guide_legend(nrow = 6))
  
  
})