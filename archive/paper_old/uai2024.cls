\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{uai2024}[2024/01/01 UAI 2024 Style]

\DeclareOption{accepted}{\gdef\@accepted{1}}
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax
\LoadClass[11pt,letterpaper]{article}

\RequirePackage[margin=1in]{geometry}
\RequirePackage{natbib}
\bibliographystyle{plainnat}

\setlength{\parskip}{0.5em}
\setlength{\parindent}{0pt}

\renewcommand{\maketitle}{%
  \begin{center}
    {\Large\bfseries\@title\par}
    \vspace{1em}
    {\large\@author\par}
    \vspace{1em}
  \end{center}
}
