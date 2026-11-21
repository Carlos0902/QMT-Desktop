using System;
using System.Diagnostics;
using System.IO;
using System.Windows.Forms;

internal static class Launcher
{
    [STAThread]
    private static void Main()
    {
        var app = AppDomain.CurrentDomain.BaseDirectory;
        var runtime = Path.Combine(app, "runtime", "pythonw.exe");
        try
        {
            var profile = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData), "QMT Desktop");
            Directory.CreateDirectory(profile);
            Process.Start(new ProcessStartInfo(runtime, "-I -X utf8 -m qmt.desktop") {
                WorkingDirectory = profile, UseShellExecute = false
            });
        }
        catch (Exception error)
        {
            MessageBox.Show(error.Message, "QMT Desktop", MessageBoxButtons.OK, MessageBoxIcon.Error);
        }
    }
}
