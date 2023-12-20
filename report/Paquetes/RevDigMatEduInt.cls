\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Paquetes/RevDigMatEduInt}[2020/10/02 Clase de la Revista Digital Matematica Educacion e Internet del Instituto Tecnologico de Costa Rica]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}
\ProcessOptions\relax

\LoadClass[a4paper, 11pt, twoside]{article}

%\RequirePackage[utf8]{inputenc}

% M\'argenes
\RequirePackage[a4paper, top=2cm, bottom=2cm, left=2cm, right=2cm, headsep=10pt]{geometry}


%\RequirePackage[a4paper, total={16cm, 22cm}, top=2cm, bottom=2cm, left=2cm, right=2cm, headsep=10pt]{geometry}

%T\'itulos en espannol
\RequirePackage[english]{babel}

%Opciones para los caption
\RequirePackage[labelfont=bf]{caption}

% Colores
\RequirePackage[table]{xcolor}

% Cabeceras bonitas
\RequirePackage{fancyhdr}

% Comandos con opciones
\RequirePackage{xargs}

% Tablas más bonitas
\RequirePackage{booktabs}

%Para opciones de los títulos de sección
\RequirePackage{titlesec}
\RequirePackage{svg}

%%Fuentes---------------------------------------------------------------
%--------------------------------------------------------%
%\RequirePackage{pslatex}                              % Fuentes finas postscript
%\RequirePackage[sc]{mathpazo}                         % Fuentes mathpazo
\RequirePackage{palatino}
\RequirePackage{tgpagella}
%\RequirePackage{eulerpx}
\RequirePackage{eucal}
%--------------------------------------------------------%
\RequirePackage[scaled=0.83]{beramono}
%\RequirePackage{chancery}
\RequirePackage{helvet}
%%-------------------------------------------------------------------

\let\openbox\relax
% Paquetes de la ams
\RequirePackage{amsthm}

\RequirePackage{xparse}					% \NewDocumentEnvironment
\RequirePackage{tcolorbox} 
\tcbuselibrary{skins,breakable}     % Librer\'ias de tcolorbox

\RequirePackage{tikz}
\usetikzlibrary{positioning,shadows,backgrounds,calc}%


%自定义
\RequirePackage{ctex}
\RequirePackage{fontspec}


%Color para los t\'itulos de las secciones y subsecciones
\definecolor{azulEntornos}{rgb}{.0,.0,.3}
\definecolor{azulTitulos}{RGB}{37,74,165}
\definecolor{amarilloEntornos}{RGB}{248,248,245} 
\definecolor{amarilloLema}{RGB}{255,202,0}
\definecolor{azulNota}{RGB}{68,0,170}
\definecolor{amarilloCaja}{RGB}{251,237,121}
\definecolor{fntGris}{rgb}{.3,.3,.3}
\definecolor{fntVerde}{RGB}{0,50,0}

% Sin identaci\'on y dejando espacio entre p\'arrafos
\setlength{\parindent}{0pt}
\setlength{\parskip}{10pt}

% Define algunos comandos para las fuentes
\newcommand{\fhvb}[2]{{\fontfamily{phv}\fontseries{b}\fontsize{#1}{1}\selectfont{#2}}}
\newcommand{\fhv}[2]{{\fontfamily{phv}\fontsize{#1}{1}\selectfont{#2}}}
\newcommandx{\fntg}[4][1=pag,2=9,3=n]{{\color{fntGris}\fontfamily{#1}\fontsize{#2}{1}\fontshape{#3}\selectfont{#4}}}
\newcommandx{\fnte}[4][1=phv,2=9,3=n]{{\color{fntVerde}\fontfamily{#1}\fontsize{#2}{1}\fontshape{#3}\selectfont{#4}}}

%TITULO - FECHA - AUTORES - RESUMEN - ABSTRACT - KEYWORDS - etc. 

% TITULO
\newcommand{\titulo}[6]{%espanol-ingl\'es-volumen-n\'umero-mes de inicio[, anno]-mes final, anno
	\title{
		\thispagestyle{empty}
		\headerInicial{V\lowercase{ol #3, }N\lowercase{o #4. }   #5 $ - $  #6}
		\begin{center}
			{\Large\bfseries #1}\\[0.22cm]
		\end{center}
		%\revistaheaderA{V\lowercase{ol #3, }N\lowercase{o #4. }  #5 $ - $  #6}
		\medskip
	}
}
%--------------------------------------------------------------------------

\newcommand{\fecha}{%
	\date{
		{\textcolor{lightgray}{\rule[0.5cm]{0.8\textwidth}{0.1pt}}} \\[-0.2cm]
		\begin{minipage}{0.6\textwidth}
			\begin{center}
			\fhv{12}{\textcolor[HTML]{7F3CFF}{\today} }  \\[0.04cm]
			\end{center}
			{\textcolor{lightgray}{\rule[0.5cm]{\textwidth}{0.1pt}}}
		\end{minipage}
		\vspace{-5mm}
	}
}

\newcommand{\unautor}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\author{
		\fnte[phv][12]{\href{https://orcid.org/#5}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} #1}\\
		\fntg[phv][10]{#2}\\
		\fntg[phv][9]{#3} \\
		\fntg[phv][10]{#4} \\
	}
}

\newcommand{\autor}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautoruno{#1}
	\def\correoautoruno{#2}
	\def\universidadautoruno{#3}
	\def\paisautoruno{#4}
	\def\orcidautoruno{#5}
}

\newcommand{\autordos}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautordos{#1}
	\def\correoautordos{#2}
	\def\universidadautordos{#3}
	\def\paisautordos{#4}
	\def\orcidautordos{#5}
}

\newcommand{\autortres}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautortres{#1}
	\def\correoautortres{#2}
	\def\universidadautortres{#3}
	\def\paisautortres{#4}
	\def\orcidautortres{#5}
}

\newcommand{\autorcuatro}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautorcuatro{#1}
	\def\correoautorcuatro{#2}
	\def\universidadautorcuatro{#3}
	\def\paisautorcuatro{#4}
	\def\orcidautorcuatro{#5}
}

\newcommand{\autorcinco}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautorcinco{#1}
	\def\correoautorcinco{#2}
	\def\universidadautorcinco{#3}
	\def\paisautorcinco{#4}
	\def\orcidautorcinco{#5}
}

\newcommand{\autorseis}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautorseis{#1}
	\def\correoautorseis{#2}
	\def\universidadautorseis{#3}
	\def\paisautorseis{#4}
	\def\orcidautorseis{#5}
}

\newcommand{\autorsiete}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautorsiete{#1}
	\def\correoautorsiete{#2}
	\def\universidadautorsiete{#3}
	\def\paisautorsiete{#4}
	\def\orcidautorsiete{#5}
}

\newcommand{\autorocho}[5]{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\def\nombreautorocho{#1}
	\def\correoautorocho{#2}
	\def\universidadautorocho{#3}
	\def\paisautorocho{#4}
	\def\orcidautorocho{#5}
}

\newcommand{\autores}{% Nombre - email - Universidad - Pa\'is - N\'umero de ORCID
	\author{
		\fnte[phv][12]{\href{https://orcid.org/\orcidautoruno}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautoruno}\\
		\fntg[phv][10]{\correoautoruno}\\
		\fntg[phv][9]{\universidadautoruno} \\
		\fntg[phv][10]{\paisautoruno} \\
		\ifx\nombreautordos\undefined\else
			\and
			\fnte[phv][12]{\href{https://orcid.org/\orcidautordos}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautordos}\\
			\fntg[phv][10]{\correoautordos}\\
			\fntg[phv][9]{\universidadautordos} \\
			\fntg[phv][10]{\paisautordos} \\
			\ifx\nombreautortres\undefined\else
				\and
				\fnte[phv][12]{\href{https://orcid.org/\orcidautortres}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautortres}\\
				\fntg[phv][10]{\correoautortres}\\
				\fntg[phv][9]{\universidadautortres} \\
				\fntg[phv][10]{\paisautortres} \\
				\ifx\nombreautorcuatro\undefined\else
					\and
					\fnte[phv][12]{\href{https://orcid.org/\orcidautorcuatro}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautorcuatro}\\
					\fntg[phv][10]{\correoautorcuatro}\\
					\fntg[phv][9]{\universidadautorcuatro} \\
					\fntg[phv][10]{\paisautorcuatro} \\
					\ifx\nombreautorcinco\undefined\else
						\and
						\fnte[phv][12]{\href{https://orcid.org/\orcidautorcinco}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautorcinco}\\
						\fntg[phv][10]{\correoautorcinco}\\
						\fntg[phv][9]{\universidadautorcinco} \\
						\fntg[phv][10]{\paisautorcinco} \\
						\ifx\nombreautorseis\undefined\else
							\and
							\fnte[phv][12]{\href{https://orcid.org/\orcidautorseis}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautorseis}\\
							\fntg[phv][10]{\correoautorseis}\\
							\fntg[phv][9]{\universidadautorseis} \\
							\fntg[phv][10]{\paisautorseis} \\
							\ifx\nombreautorsiete\undefined\else
								\and
								\fnte[phv][12]{\href{https://orcid.org/\orcidautorsiete}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautorsiete}\\
								\fntg[phv][10]{\correoautorsiete}\\
								\fntg[phv][9]{\universidadautorsiete} \\
								\fntg[phv][10]{\paisautorsiete} \\
								\ifx\nombreautorocho\undefined\else
									\and
									\fnte[phv][12]{\href{https://orcid.org/\orcidautorocho}{\includegraphics[scale=0.65]{./images/ORCID-iD_icon-16x16.png}} \nombreautorocho}\\
									\fntg[phv][10]{\correoautorocho}\\
									\fntg[phv][9]{\universidadautorocho} \\
									\fntg[phv][10]{\paisautorocho} \\
								\fi
							\fi
						\fi
					\fi
				\fi
			\fi
		\fi
	}
}

\newenvironment{resumen}{%
	{\bfseries \color{azulEntornos} Resumen: }%
}

\newenvironment{palabrasclave}{%
	{\bfseries \color{azulEntornos} Palabras Clave: }%
}

\newenvironment{keywords}{%
	{\bfseries \color{azulEntornos} Keywords: }%
}

\renewenvironment{abstract}{%
	{\bfseries \color{azulEntornos} Abstract: }%
}

%********************************** Bibliografía **********************

\makeatletter
\renewenvironment{thebibliography}[1]
{\section{\bibname}% <-- this line was changed from \chapter* to \section*
	\@mkboth{\MakeUppercase\bibname}{\MakeUppercase\bibname}%
	\list{\@biblabel{\@arabic\c@enumiv}}%
	{\settowidth\labelwidth{\@biblabel{#1}}%
		\leftmargin\labelwidth
		\advance\leftmargin\labelsep
		\@openbib@code
		\usecounter{enumiv}%
		\let\p@enumiv\@empty
		\renewcommand\theenumiv{\@arabic\c@enumiv}}%
	\sloppy
	\clubpenalty4000
	\@clubpenalty \clubpenalty
	\widowpenalty4000%
	\sfcode`\.\@m}
{\def\@noitemerr
	{\@latex@warning{Empty `thebibliography' environment}}%
	\endlist}
\makeatother


%********************************** DISENO ****************************

%-----------------------------------------------------------------------
% Cabeceras
%-----------------------------------------------------------------------

\pagestyle{fancy} % Estilo de la p\'agina fancy (usa fancyhdr)
\newcommand{\inforevista}{\protect{\printoffprintinfo}}
\setlength{\headsep}{0.2in} %separaci\'on header con el texto
\newcommand{\helv}{\fontfamily{phv}\fontsize{7.5}{10}\selectfont}
\renewcommand{\sectionmark}[1]{{\markright{#1}{}}}
\fancyhf{} % borra cabecera y pie actuales
\fancyhead[LE,RO]{\bfseries \helv\thepage} %Left Even page - Right Odd page
%\fancyhead[R]{\bfseries \helv\thepage}  % Numeraci\'on siempre a la derecha
\fancyhead[LO]{\helv\rightmark}
\fancyhead[RE]{\helv\leftmark}
\fancyfoot{} % clear all footer fields
\fancyfoot[L]{{\textcolor{lightgray}{\rule[0.5cm]{0.1\textwidth}{0.1pt}}}\* \\[-0.55cm]\inforevista}
\renewcommand{\headrulewidth}{0pt} % Sin raya. Con raya?: cambiar {0} por {0.5pt}
\renewcommand{\footrulewidth}{0pt}
\setlength\headheight{8.5pt}
\fancyheadoffset[R]{0.0cm} %Numeraci\'on de p\'agina en el borde de la p\'agina
\addtolength{\headheight}{0.5pt} % espacio para la raya
\fancypagestyle{plain}{%
\fancyhead{} % elimina cabeceras y raya en p\'aginas "plain"
\renewcommand{\headrulewidth}{0pt} }
% Fin cabeceras ---------------------------------------------


%%-------------------------------------------------------------
%%	Numeración y color de los títulos de las secciones
%%-------------------------------------------------------------

\titleformat{\section}{\color{azulTitulos}\normalfont\Large\sffamily\bfseries}{\thesection.}{1em}{}[\medskip{\titlerule[0.2pt]}]

\titleformat{\subsection}{\color{azulTitulos}\normalfont\large\sffamily\bfseries}{\thesubsection.}{1em}{}

\titleformat{\subsubsection}{\color{azulTitulos}\normalfont\normalsize\sffamily\bfseries}{\thesubsubsection.}{1em}{}

% Fin numeraci***n secciones ----------------------------------------------------

%%---------------------------------
%% Imprimir info al pie de p\'agina
%%---------------------------------
	\newif\ifoffprintinfo
	\def\dooffprintinfo{\global\offprintinfotrue}
	
	\def\copyrightyear#1{\def\thecopyrightyear{#1}}
	
	\copyrightyear{\the\year}
	
	\def\dofnote#1#2{\vtop{\hyphenpenalty=10000
	\advance\hsize -10pt \raggedright
	\footnotesize{\it #1. }{ #2}\\
	}}
	\def\offprintinfo#1#2{
	\def\theoffprint{\bgroup\frenchspacing
	\dofnote{#1}{#2}
	\egroup}}
	
	\def\x@makefntext#1{
	\kern-3\p@
	\hrule\@width.4\columnwidth
	\kern2.6\p@
	\vrule height 9pt width0pt \relax
	#1}
	
	\def\offprintinfoerror{\typeout{^^J^^J
	!! Debe poner {\string\offprintinfo\string{(Title,
	Edition)\string}\string{(Author)\string}^^J en el inicio del documento.!!^^J^^J}}
	\bgroup
	\x@makefntext{Debe poner {\tt \string\offprintinfo\string{(Title,
	Edition)\string}\string{(Author)\string}\newline en el inicio del documento.\vrule depth8pt width0pt}\egroup}}
	
	
	\def\printoffprintinfo{\vtop to0pt{%
	\hsize=\textwidth\footnotesize
	\expandafter\ifx\csname theoffprint\endcsname\relax
	\offprintinfoerror\else\theoffprint\fi\vskip1sp\vss}}
	
	
	\def\@xfootnote[#1]{%
	  \protected@xdef\@thefnmark{#1}%
	  \@footnotemark\@footnotetext}
% Fin Imprimir info al pie de p\'agina

\newcommand{\headerInicial}[1]{
	\vspace{-2cm}
	\textcolor{lightgray}{\rule[0.5cm]{\textwidth}{4pt}}\\[-0.2cm]
	\begin{minipage}{0.4\textwidth}
		\includesvg[width=6cm]{images/logo.svg}
		\vspace{10pt}
	\end{minipage}
	\begin{minipage}{0.45\textwidth}
		\vspace{10pt}
		\fhvb{10}{\textcolor[HTML]{7F3CFF}{Pro.多元统计分析}}\\[-0.3cm]
		\fhvb{10}{\textcolor[HTML]{7F3CFF}{指导教师:李野}}\\[-0.3cm]
		\fntg[phv][9]{\bfseries\textcolor[HTML]{7F3CFF}{仓库地址:}
		\href{https://github.com/DGMEFG/anime_cls}{\textcolor[HTML]{7F3CFF}{https://github.com/DGMEFG/anime\_cls}}}\\[-0.3cm]
	\end{minipage}
	\textcolor{lightgray}{\rule[0.5cm]{\textwidth}{0.1pt}}
}

%\revistaheaderA{V\lowercase{ol 12, }N\lowercase{o 2. }  M\lowercase{arzo} $-$ A\lowercase{gosto} 2012.}
\newcommand{\revistaheaderA}[1]{\markboth{R\lowercase{evista digital} M\lowercase{atem\'atica,} E\lowercase{ducaci\'on e }
I\lowercase{nternet} \lowercase{\href{https://revistas.tec.ac.cr/index.php/matematica}{\lowercase{(https://revistas.tec.ac.cr/index.php/matematica)}}}.
#1}{R\lowercase{evista digital} M\lowercase{atem\'atica,} E\lowercase{ducaci\'on e }
I\lowercase{nternet} \lowercase{\href{https://revistas.tec.ac.cr/index.php/matematica}{\lowercase{(https://revistas.tec.ac.cr/index.php/matematica)}}}.
#1}}