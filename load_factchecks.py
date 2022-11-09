# -*- coding: utf-8 -*-
import loaders.googlefactcheck

checker=loaders.googlefactcheck.GoogleFactCheckLoader()
checker.loadFactChecks()
checker.update_last_run()
