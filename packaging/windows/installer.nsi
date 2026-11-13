; LLM Surgery Studio: per-user Windows installer (no administrator rights).
;
; Built by packaging/build-windows.ps1:
;   makensis /DVERSION=0.1.0 /DSTAGE=<folder with the binaries> /DOUTFILE=<setup.exe> installer.nsi
;
; Everything goes under %LOCALAPPDATA% and HKCU. The user's data (components, settings,
; models) lives in %LOCALAPPDATA%\weight-surgeon and is only removed when the user says so.

Unicode true
SetCompressor /SOLID lzma
ManifestDPIAware true
RequestExecutionLevel user
SetOverwrite on

!ifndef VERSION
  !error "pass /DVERSION=x.y.z"
!endif
!ifndef STAGE
  !error "pass /DSTAGE=<folder with surgeon-gui.exe and surgeon.exe>"
!endif
!ifndef OUTFILE
  !define OUTFILE "setup.exe"
!endif

!define APP "LLM Surgery Studio"
!define UNINSTKEY "Software\Microsoft\Windows\CurrentVersion\Uninstall\LLMSurgeryStudio"

Name "${APP}"
OutFile "${OUTFILE}"
InstallDir "$LOCALAPPDATA\Programs\LLM Surgery Studio"
InstallDirRegKey HKCU "Software\LLMSurgeryStudio" "InstallDir"
VIProductVersion "${VERSION}.0"
VIAddVersionKey "ProductName" "${APP}"
VIAddVersionKey "FileDescription" "${APP} installer"
VIAddVersionKey "FileVersion" "${VERSION}"
VIAddVersionKey "ProductVersion" "${VERSION}"

!include "MUI2.nsh"
!include "LogicLib.nsh"
!include "StrFunc.nsh"
${StrLoc}
${StrRep}
${UnStrRep}

!define MUI_ICON "${NSISDIR}\Contrib\Graphics\Icons\modern-install.ico"
!define MUI_ABORTWARNING
!define MUI_FINISHPAGE_RUN "$INSTDIR\surgeon-gui.exe"
!define MUI_FINISHPAGE_RUN_TEXT "Lancer ${APP} / Launch ${APP}"

!insertmacro MUI_PAGE_WELCOME
!insertmacro MUI_PAGE_COMPONENTS
!insertmacro MUI_PAGE_DIRECTORY
!insertmacro MUI_PAGE_INSTFILES
!insertmacro MUI_PAGE_FINISH
!insertmacro MUI_UNPAGE_CONFIRM
!insertmacro MUI_UNPAGE_INSTFILES
!insertmacro MUI_LANGUAGE "French"
!insertmacro MUI_LANGUAGE "English"

; A running executable cannot be overwritten, but it can be renamed: the old one is moved
; aside and removed at the next install or uninstall.
!macro ReplaceFile NAME
  IfFileExists "$INSTDIR\${NAME}.old" 0 +2
    Delete "$INSTDIR\${NAME}.old"
  IfFileExists "$INSTDIR\${NAME}" 0 +3
    ClearErrors
    Rename "$INSTDIR\${NAME}" "$INSTDIR\${NAME}.old"
  File "${STAGE}\${NAME}"
!macroend

Section "Application" SecApp
  SectionIn RO
  SetOutPath "$INSTDIR"
  !insertmacro ReplaceFile "surgeon-gui.exe"
  !insertmacro ReplaceFile "surgeon.exe"
  File "${STAGE}\README-WINDOWS.txt"
  WriteUninstaller "$INSTDIR\uninstall.exe"
  WriteRegStr HKCU "Software\LLMSurgeryStudio" "InstallDir" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayName" "${APP}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayVersion" "${VERSION}"
  WriteRegStr HKCU "${UNINSTKEY}" "DisplayIcon" "$INSTDIR\surgeon-gui.exe"
  WriteRegStr HKCU "${UNINSTKEY}" "Publisher" "weight-surgeon"
  WriteRegStr HKCU "${UNINSTKEY}" "InstallLocation" "$INSTDIR"
  WriteRegStr HKCU "${UNINSTKEY}" "UninstallString" '"$INSTDIR\uninstall.exe"'
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoModify" 1
  WriteRegDWORD HKCU "${UNINSTKEY}" "NoRepair" 1
SectionEnd

Section "Raccourci du menu Démarrer / Start menu shortcut" SecStart
  CreateShortCut "$SMPROGRAMS\${APP}.lnk" "$INSTDIR\surgeon-gui.exe"
SectionEnd

Section /o "Raccourci du bureau / Desktop shortcut" SecDesktop
  CreateShortCut "$DESKTOP\${APP}.lnk" "$INSTDIR\surgeon-gui.exe"
SectionEnd

; Optional and off by default: adds the install folder (with surgeon.exe) to the user PATH.
Section /o "Ajouter surgeon.exe au PATH / Add surgeon.exe to PATH" SecPath
  ReadRegStr $0 HKCU "Environment" "Path"
  ${StrLoc} $1 ";$0;" ";$INSTDIR;" ">"
  ${If} $1 == ""
    ${If} $0 == ""
      StrCpy $0 "$INSTDIR"
    ${Else}
      StrCpy $0 "$0;$INSTDIR"
    ${EndIf}
    WriteRegExpandStr HKCU "Environment" "Path" "$0"
    SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
  ${EndIf}
SectionEnd

Section "Uninstall"
  Delete "$SMPROGRAMS\${APP}.lnk"
  Delete "$DESKTOP\${APP}.lnk"

  ; Take the folder out of the user PATH, whether or not the section was selected.
  ReadRegStr $0 HKCU "Environment" "Path"
  ${If} $0 != ""
    StrCpy $1 ";$0;"
    ${UnStrRep} $1 "$1" ";$INSTDIR;" ";"
    StrCpy $1 $1 -1
    StrCpy $1 $1 "" 1
    ${If} $1 != $0
      WriteRegExpandStr HKCU "Environment" "Path" "$1"
      SendMessage ${HWND_BROADCAST} ${WM_WININICHANGE} 0 "STR:Environment" /TIMEOUT=5000
    ${EndIf}
  ${EndIf}

  Delete "$INSTDIR\surgeon-gui.exe"
  Delete "$INSTDIR\surgeon-gui.exe.old"
  Delete "$INSTDIR\surgeon.exe"
  Delete "$INSTDIR\surgeon.exe.old"
  Delete "$INSTDIR\README-WINDOWS.txt"
  Delete "$INSTDIR\uninstall.exe"
  RMDir "$INSTDIR"
  DeleteRegKey HKCU "${UNINSTKEY}"
  DeleteRegKey HKCU "Software\LLMSurgeryStudio"

  ; The user's models, components and settings are theirs: kept unless they ask.
  IfSilent keep
  MessageBox MB_YESNO|MB_DEFBUTTON2 "Supprimer aussi les composants installés, les réglages et les journaux ($LOCALAPPDATA\weight-surgeon) ?$\nVos modèles et workspaces ne sont jamais touchés.$\n$\nAlso delete the installed components, settings and logs?" IDNO keep
  RMDir /r "$LOCALAPPDATA\weight-surgeon"
  keep:
SectionEnd
