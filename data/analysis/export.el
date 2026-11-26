;; Run every R block of the org file given on the command line, save it, export to PDF.
;; Usage: /Applications/Emacs.app/Contents/MacOS/Emacs --batch -l export.el resultados.org
(require 'org)
(require 'ob-R)
(require 'ox-latex)
(org-babel-do-load-languages 'org-babel-load-languages '((R . t)))
(setq org-confirm-babel-evaluate nil
      org-export-use-babel t
      org-latex-pdf-process '("latexmk -pdf -interaction=nonstopmode -output-directory=%o %f"))
(let ((file (expand-file-name (car command-line-args-left))))
  (setq command-line-args-left nil)
  (find-file file)
  (org-babel-execute-buffer)
  (save-buffer)
  (org-latex-export-to-pdf))
