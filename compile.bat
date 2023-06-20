nasm -fwin32 test.asm
link /nodefaultlib /entry:main /subsystem:console test.obj "C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22000.0\um\x86\kernel32.lib"
