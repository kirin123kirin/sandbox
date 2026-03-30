#Requires -Version 5.1
<#
.SYNOPSIS
    個人情報・機密データのマスキングツール

.DESCRIPTION
    テキストファイル・Office ファイル（Word/Excel/PowerPoint）内の
    個人情報および機密データを検出してマスキングします。

    対応マスキング項目:
        [個人情報]
          -Email       : メールアドレス
          -Phone       : 電話番号（固定・携帯・フリーダイヤル）
          -CreditCard  : クレジットカード番号
          -Address     : 住所・郵便番号
          -Name        : 氏名（ラベル付き・敬称付き・マイナンバー）

        [機密データ]
          -Confidential: APIキー・パスワード・JWT・AWSキー・GitHubトークン
                         プライベートキー・IPアドレス・接続文字列

    対応ファイル形式:
        テキスト系: .txt .csv .tsv .log .json .xml .html .htm
                    .md .yaml .yml .ini .conf .config .sql .ps1 .sh
        Office 系 : .docx .doc .docm  /  .xlsx .xls .xlsm
                    .pptx .ppt .pptm
        ※ Office 処理には Microsoft Office のインストールが必要です。

.PARAMETER Path
    マスキング対象のファイルまたはディレクトリパス（必須）

.PARAMETER OutputPath
    マスキング後の出力先（省略時は元ファイルを上書き・バックアップあり）

.PARAMETER All
    全パターンでマスキング（デフォルト動作）

.PARAMETER Email
    メールアドレスのみマスキング

.PARAMETER Phone
    電話番号のみマスキング

.PARAMETER CreditCard
    クレジットカード番号のみマスキング

.PARAMETER Address
    住所・郵便番号のみマスキング

.PARAMETER Name
    氏名・マイナンバーのみマスキング

.PARAMETER Confidential
    機密データのみマスキング

.PARAMETER KnownNames
    マスキングする既知の名前リスト（例: @("山田太郎","田中花子")）

.PARAMETER MaskChar
    マスキング文字（デフォルト: *）

.PARAMETER DryRun
    実際には変更せず、検出内容のみ表示

.PARAMETER Recurse
    ディレクトリを再帰的に処理

.EXAMPLE
    # 単一ファイルを全パターンでマスキング（出力先指定）
    .\Invoke-Masking.ps1 -Path "C:\Docs\report.docx" -All -OutputPath "C:\Masked\"

.EXAMPLE
    # ディレクトリをメール・電話番号のみ再帰処理
    .\Invoke-Masking.ps1 -Path "C:\Data\" -Email -Phone -Recurse

.EXAMPLE
    # 既知の名前を指定してドライラン（検出確認）
    .\Invoke-Masking.ps1 -Path "C:\Data\data.csv" -All -KnownNames @("山田太郎","田中花子") -DryRun

.EXAMPLE
    # 機密情報のみマスキング
    .\Invoke-Masking.ps1 -Path "C:\src\" -Confidential -Recurse -MaskChar "#"
#>

[CmdletBinding(SupportsShouldProcess)]
param(
    [Parameter(Mandatory, Position = 0, HelpMessage = 'マスキング対象のファイルまたはディレクトリ')]
    [string]$Path,

    [Parameter(HelpMessage = '出力先パス（省略時は上書き）')]
    [string]$OutputPath = '',

    [switch]$All,
    [switch]$Email,
    [switch]$Phone,
    [switch]$CreditCard,
    [switch]$Address,
    [switch]$Name,
    [switch]$Confidential,

    [string[]]$KnownNames = @(),

    [string]$MaskChar = '*',

    [switch]$DryRun,
    [switch]$Recurse
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

# ==============================================================================
#  ロギング
# ==============================================================================
function Write-MaskLog {
    param(
        [string]$Message,
        [ValidateSet('INFO','WARN','ERROR','DEBUG')]
        [string]$Level = 'INFO'
    )
    $ts = Get-Date -Format 'HH:mm:ss'
    $line = "[$ts][$Level] $Message"
    switch ($Level) {
        'ERROR' { Write-Host $line -ForegroundColor Red }
        'WARN'  { Write-Host $line -ForegroundColor Yellow }
        'DEBUG' { Write-Verbose $line }
        default { Write-Host $line }
    }
}

# ==============================================================================
#  カテゴリ有効判定
# ==============================================================================
function Test-CategoryEnabled {
    param([string]$Category)
    $useAll = $All -or (-not ($Email -or $Phone -or $CreditCard -or $Address -or $Name -or $Confidential))
    switch ($Category) {
        'Email'        { return ($useAll -or $Email.IsPresent) }
        'Phone'        { return ($useAll -or $Phone.IsPresent) }
        'CreditCard'   { return ($useAll -or $CreditCard.IsPresent) }
        'Address'      { return ($useAll -or $Address.IsPresent) }
        'Name'         { return ($useAll -or $Name.IsPresent) }
        'Confidential' { return ($useAll -or $Confidential.IsPresent) }
    }
    return $false
}

# ==============================================================================
#  マスク変換ヘルパー
# ==============================================================================
function Get-MaskString { param([int]$Len) return $MaskChar * [Math]::Max(1, $Len) }

function ConvertTo-MaskedEmail {
    param([string]$Value)
    $at = $Value.IndexOf('@')
    if ($at -le 0) { return Get-MaskString $Value.Length }
    $local  = $Value.Substring(0, $at)
    $domain = $Value.Substring($at)   # @domain.tld はそのまま残す
    $maskedLocal = if ($local.Length -le 2) {
        Get-MaskString $local.Length
    } else {
        "$($local[0])$(Get-MaskString ($local.Length - 2))$($local[-1])"
    }
    return "$maskedLocal$domain"
}

function ConvertTo-MaskedPhone {
    param([string]$Value)
    $digits = $Value -replace '\D', ''
    if ($digits.Length -lt 7) { return Get-MaskString $Value.Length }
    # 末尾4桁以外をマスク（元フォーマット維持）
    $sb = [System.Text.StringBuilder]::new()
    $idx = 0
    foreach ($c in $Value.ToCharArray()) {
        if ($c -match '\d') {
            $null = $sb.Append($(if ($idx -lt ($digits.Length - 4)) { $MaskChar } else { $c }))
            $idx++
        } else {
            $null = $sb.Append($c)
        }
    }
    return $sb.ToString()
}

function ConvertTo-MaskedCreditCard {
    param([string]$Value)
    $digits = $Value -replace '\D', ''
    if ($digits.Length -lt 12) { return Get-MaskString $Value.Length }
    # 先頭6桁 + マスク + 末尾4桁（元フォーマット維持）
    $sb = [System.Text.StringBuilder]::new()
    $idx = 0
    foreach ($c in $Value.ToCharArray()) {
        if ($c -match '\d') {
            $visible = ($idx -lt 6) -or ($idx -ge $digits.Length - 4)
            $null = $sb.Append($(if ($visible) { $c } else { $MaskChar }))
            $idx++
        } else {
            $null = $sb.Append($c)
        }
    }
    return $sb.ToString()
}

function ConvertTo-MaskedIP {
    param([string]$Value)
    $parts = $Value -split '\.'
    if ($parts.Count -ne 4) { return Get-MaskString $Value.Length }
    # 第1・第2オクテットを残し第3・第4をマスク
    return "$($parts[0]).$($parts[1]).$(Get-MaskString $parts[2].Length).$(Get-MaskString $parts[3].Length)"
}

# ==============================================================================
#  パターン定義
#  各エントリ:
#    Category  - カテゴリ名
#    Desc      - 説明（ログ表示用）
#    Pattern   - 正規表現
#    IgnoreCase- 大文字小文字を無視するか
#    Key       - 置換ロジックの識別キー
# ==============================================================================
function Get-MaskPatterns {
    return @(
        # ---- 個人情報 --------------------------------------------------------
        @{ Category='Email';       Desc='メールアドレス';             IgnoreCase=$false; Key='Email'
           Pattern='[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}' }

        @{ Category='Phone';       Desc='電話番号';                   IgnoreCase=$false; Key='Phone'
           Pattern='(?<!\d)(?:0\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{3,4}|0[789]0[-.\s]?\d{4}[-.\s]?\d{4}|0120[-.\s]?\d{3}[-.\s]?\d{3})(?!\d)' }

        @{ Category='CreditCard';  Desc='クレジットカード番号';        IgnoreCase=$false; Key='CreditCard'
           Pattern='(?<!\d)(?:\d{4}[-\s]?){3}\d{4}(?!\d)' }

        @{ Category='Address';     Desc='郵便番号';                   IgnoreCase=$false; Key='PostalCode'
           Pattern='〒?\d{3}[-－]\d{4}' }

        @{ Category='Address';     Desc='住所';                       IgnoreCase=$false; Key='Address'
           Pattern='(?:北海道|青森|岩手|宮城|秋田|山形|福島|茨城|栃木|群馬|埼玉|千葉|東京|神奈川|新潟|富山|石川|福井|山梨|長野|岐阜|静岡|愛知|三重|滋賀|京都|大阪|兵庫|奈良|和歌山|鳥取|島根|岡山|広島|山口|徳島|香川|愛媛|高知|福岡|佐賀|長崎|熊本|大分|宮崎|鹿児島|沖縄)(?:都|道|府|県)?[^\r\n]{5,60}?(?:\d+丁目|\d+番(?:地|\d*号)|\d+[-－]\d+[-－]?\d*)' }

        @{ Category='Name';        Desc='マイナンバー';                IgnoreCase=$true;  Key='MyNumber'
           Pattern='(?:マイナンバー|個人番号|my.?number)[^\d]{0,5}(\d{4}[- ]?\d{4}[- ]?\d{4})' }

        @{ Category='Name';        Desc='氏名（ラベル付き）';          IgnoreCase=$false; Key='NameLabel'
           Pattern='(?:氏名|お名前|名前|ご氏名|担当者名?|申込者名?|契約者名?|被保険者名?|代表者名?|送付先名?|請求先名?|宛名)[：:　\s]{0,3}([^\r\n、。,\s]{2,20})' }

        @{ Category='Name';        Desc='氏名（敬称付き）';            IgnoreCase=$false; Key='NameHonorific'
           Pattern='[\p{IsCJKUnifiedIdeographs}\p{IsKatakana}\p{IsHiragana}a-zA-Z]{2,10}(?:様|さん|先生|殿|氏)(?=[\s、。\r\n]|$)' }

        # ---- 機密情報 --------------------------------------------------------
        @{ Category='Confidential'; Desc='AWSアクセスキー';            IgnoreCase=$false; Key='AWSKey'
           Pattern='(?:AKIA|ASIA|AROA|ANPA|ANVA|APKA)[A-Z0-9]{16}' }

        @{ Category='Confidential'; Desc='GitHubトークン';             IgnoreCase=$false; Key='GitHubToken'
           Pattern='(?:ghp|ghs|gho|ghu|ghr)_[A-Za-z0-9]{36}' }

        @{ Category='Confidential'; Desc='JWTトークン';                IgnoreCase=$false; Key='JWT'
           Pattern='eyJ[A-Za-z0-9_\-]{10,}\.eyJ[A-Za-z0-9_\-]{10,}\.[A-Za-z0-9_\-]+' }

        @{ Category='Confidential'; Desc='プライベートキー';            IgnoreCase=$false; Key='PrivateKey'
           Pattern='-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z ]+ )?PRIVATE KEY-----' }

        @{ Category='Confidential'; Desc='APIキー/トークン';           IgnoreCase=$true;  Key='APIKey'
           Pattern='(?:api[_\-]?key|api[_\-]?secret|access[_\-]?token|auth[_\-]?token|secret[_\-]?key|bearer)[\x22\x27\s:=]+[A-Za-z0-9+/=_\-]{20,}' }

        @{ Category='Confidential'; Desc='パスワード';                  IgnoreCase=$true;  Key='Password'
           Pattern="(?:password|passwd|pwd|pass)[\x22\x27\s:=]+[^\s\x22\x27\r\n]{6,}" }

        @{ Category='Confidential'; Desc='接続文字列（認証情報）';      IgnoreCase=$true;  Key='ConnStr'
           Pattern='(?:Server|Data Source|Host)=[^;]+;[^;]*(?:Password|Pwd)=[^;\x22\x27\s]+' }

        @{ Category='Confidential'; Desc='IPアドレス';                  IgnoreCase=$false; Key='IPAddress'
           Pattern='\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b' }
    )
}

# ==============================================================================
#  マッチした値を置換文字列へ変換
# ==============================================================================
function Invoke-MaskReplace {
    param([string]$Key, [string]$MatchedValue)
    switch ($Key) {
        'Email'        { return ConvertTo-MaskedEmail $MatchedValue }
        'Phone'        { return ConvertTo-MaskedPhone $MatchedValue }
        'CreditCard'   { return ConvertTo-MaskedCreditCard $MatchedValue }
        'PostalCode'   { return ($MatchedValue -replace '\d', $MaskChar) }
        'Address'      { return '***住所マスク済***' }
        'MyNumber'     { return [regex]::Replace($MatchedValue, '\d', $MaskChar) }
        'NameLabel'    {
            if ($MatchedValue -match '([^\r\n、。,\s]{2,20})$') {
                $n = $Matches[1]
                return $MatchedValue -replace ([regex]::Escape($n)), (Get-MaskString $n.Length)
            }
            return $MatchedValue
        }
        'NameHonorific' {
            if ($MatchedValue -match '^(.{2,10})(様|さん|先生|殿|氏)$') {
                return (Get-MaskString $Matches[1].Length) + $Matches[2]
            }
            return Get-MaskString $MatchedValue.Length
        }
        'AWSKey'       { return '***AWS_ACCESS_KEY***' }
        'GitHubToken'  { return '***GITHUB_TOKEN***' }
        'JWT'          { return '***JWT_TOKEN***' }
        'PrivateKey'   { return "-----BEGIN PRIVATE KEY-----`n***PRIVATE_KEY_MASKED***`n-----END PRIVATE KEY-----" }
        'APIKey'       { return [regex]::Replace($MatchedValue, '(?i)(?<=(?:api[_\-]?key|api[_\-]?secret|access[_\-]?token|auth[_\-]?token|secret[_\-]?key|bearer)[\x22\x27\s:=]+)[A-Za-z0-9+/=_\-]{20,}', '***API_KEY***') }
        'Password'     { return [regex]::Replace($MatchedValue, "(?i)(?<=(?:password|passwd|pwd|pass)[\x22\x27\s:=]+)[^\s\x22\x27\r\n]{6,}", '***PASSWORD***') }
        'ConnStr'      { return [regex]::Replace($MatchedValue, '(?i)(?<=(?:Password|Pwd)=)[^;\x22\x27\s]+', '***PASSWORD***') }
        'IPAddress'    { return ConvertTo-MaskedIP $MatchedValue }
        default        { return Get-MaskString $MatchedValue.Length }
    }
}

# ==============================================================================
#  テキスト文字列マスキング（コア処理）
# ==============================================================================
function Invoke-TextMasking {
    param(
        [Parameter(Mandatory, ValueFromPipeline)]
        [AllowEmptyString()]
        [string]$Text
    )

    if ([string]::IsNullOrEmpty($Text)) { return $Text }

    $result  = $Text
    $detected = [System.Collections.Generic.List[string]]::new()

    # 1) 既知の名前（最優先）
    if (Test-CategoryEnabled 'Name') {
        foreach ($name in $KnownNames) {
            if ([string]::IsNullOrWhiteSpace($name)) { continue }
            $escaped = [regex]::Escape($name.Trim())
            if ([regex]::IsMatch($result, $escaped)) {
                $detected.Add("氏名(指定): $name")
                if (-not $DryRun) {
                    $result = [regex]::Replace($result, $escaped, (Get-MaskString $name.Length))
                }
            }
        }
    }

    # 2) パターンマスキング（後ろから置換してインデックスずれを防ぐ）
    foreach ($p in Get-MaskPatterns) {
        if (-not (Test-CategoryEnabled $p.Category)) { continue }

        try {
            $rxOpts = [System.Text.RegularExpressions.RegexOptions]::Multiline
            if ($p.IgnoreCase) {
                $rxOpts = $rxOpts -bor [System.Text.RegularExpressions.RegexOptions]::IgnoreCase
            }
            $rx = [regex]::new($p.Pattern, $rxOpts)
            $matches = $rx.Matches($result)
            if ($matches.Count -eq 0) { continue }

            $preview = $matches[0].Value
            if ($preview.Length -gt 40) { $preview = $preview.Substring(0, 37) + '...' }
            $detected.Add("$($p.Desc): $preview")

            if (-not $DryRun) {
                # 後ろから置換してインデックスずれを防ぐ
                $matchArr = @($matches)
                [Array]::Reverse($matchArr)
                foreach ($m in $matchArr) {
                    $replacement = Invoke-MaskReplace -Key $p.Key -MatchedValue $m.Value
                    $result = $result.Remove($m.Index, $m.Length).Insert($m.Index, $replacement)
                }
            }
        }
        catch {
            Write-MaskLog "パターン '$($p.Desc)' でエラー: $_" 'WARN'
        }
    }

    if ($DryRun -and $detected.Count -gt 0) {
        foreach ($d in $detected) { Write-MaskLog "    [検出] $d" }
    }

    return $result
}

# ==============================================================================
#  エンコーディング検出
# ==============================================================================
function Get-TextEncoding {
    param([string]$FilePath)
    $bytes = [System.IO.File]::ReadAllBytes($FilePath)
    if ($bytes.Length -ge 3 -and $bytes[0] -eq 0xEF -and $bytes[1] -eq 0xBB -and $bytes[2] -eq 0xBF) {
        return [System.Text.Encoding]::UTF8
    }
    if ($bytes.Length -ge 2) {
        if ($bytes[0] -eq 0xFF -and $bytes[1] -eq 0xFE) { return [System.Text.Encoding]::Unicode }
        if ($bytes[0] -eq 0xFE -and $bytes[1] -eq 0xFF) { return [System.Text.Encoding]::BigEndianUnicode }
    }
    return [System.Text.Encoding]::UTF8  # デフォルト UTF-8
}

function Resolve-DestDir {
    param([string]$DestPath)
    $dir = Split-Path $DestPath -Parent
    if ($dir -and -not (Test-Path $dir)) {
        New-Item -ItemType Directory -Path $dir -Force | Out-Null
    }
}

# ==============================================================================
#  テキストファイル処理
# ==============================================================================
function Invoke-TextFileMasking {
    param([string]$FilePath, [string]$DestPath)
    Write-MaskLog "テキスト: $(Split-Path $FilePath -Leaf)"
    $enc     = Get-TextEncoding $FilePath
    $content = [System.IO.File]::ReadAllText($FilePath, $enc)
    $masked  = Invoke-TextMasking -Text $content
    if (-not $DryRun) {
        Resolve-DestDir $DestPath
        [System.IO.File]::WriteAllText($DestPath, $masked, $enc)
        Write-MaskLog "  -> 保存: $DestPath"
    }
}

# ==============================================================================
#  Word ファイル処理
# ==============================================================================
function Invoke-WordFileMasking {
    param([string]$FilePath, [string]$DestPath)
    Write-MaskLog "Word: $(Split-Path $FilePath -Leaf)"
    $word = $null; $doc = $null
    try {
        $word = New-Object -ComObject Word.Application
        $word.Visible      = $false
        $word.DisplayAlerts = 0

        $doc = $word.Documents.Open($FilePath)

        # 本文段落
        for ($i = 1; $i -le $doc.Paragraphs.Count; $i++) {
            $r = $doc.Paragraphs.Item($i).Range
            $t = $r.Text
            if (-not [string]::IsNullOrWhiteSpace($t)) {
                $m = Invoke-TextMasking -Text $t
                if ($m -ne $t -and -not $DryRun) { $r.Text = $m }
            }
        }

        # テーブル
        foreach ($tbl in $doc.Tables) {
            foreach ($row in $tbl.Rows) {
                foreach ($cell in $row.Cells) {
                    $t = $cell.Range.Text -replace '\a$', ''
                    if (-not [string]::IsNullOrWhiteSpace($t)) {
                        $m = Invoke-TextMasking -Text $t
                        if ($m -ne $t -and -not $DryRun) { $cell.Range.Text = $m }
                    }
                }
            }
        }

        # ヘッダー・フッター
        foreach ($section in $doc.Sections) {
            $hfItems = @()
            try { $hfItems += @($section.Headers) } catch {}
            try { $hfItems += @($section.Footers) } catch {}
            foreach ($hf in $hfItems) {
                $t = $hf.Range.Text
                if (-not [string]::IsNullOrWhiteSpace($t)) {
                    $m = Invoke-TextMasking -Text $t
                    if ($m -ne $t -and -not $DryRun) { $hf.Range.Text = $m }
                }
            }
        }

        # テキストボックス・図形
        foreach ($shape in $doc.Shapes) {
            if ($shape.HasTextFrame -and $shape.TextFrame.HasText) {
                $t = $shape.TextFrame.TextRange.Text
                $m = Invoke-TextMasking -Text $t
                if ($m -ne $t -and -not $DryRun) { $shape.TextFrame.TextRange.Text = $m }
            }
        }

        if (-not $DryRun) {
            Resolve-DestDir $DestPath
            $doc.SaveAs2($DestPath)
            Write-MaskLog "  -> 保存: $DestPath"
        }
    }
    finally {
        if ($doc)  { try { $doc.Close($false)  } catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($doc)  | Out-Null }
        if ($word) { try { $word.Quit()         } catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null }
        [GC]::Collect(); [GC]::WaitForPendingFinalizers()
    }
}

# ==============================================================================
#  Excel ファイル処理
# ==============================================================================
function Invoke-ExcelFileMasking {
    param([string]$FilePath, [string]$DestPath)
    Write-MaskLog "Excel: $(Split-Path $FilePath -Leaf)"
    $excel = $null; $wb = $null
    try {
        $excel = New-Object -ComObject Excel.Application
        $excel.Visible      = $false
        $excel.DisplayAlerts = $false

        $wb = $excel.Workbooks.Open($FilePath)

        foreach ($ws in $wb.Worksheets) {
            Write-MaskLog "  シート: $($ws.Name)"
            $used = $ws.UsedRange
            if (-not $used) { continue }
            for ($r = 1; $r -le $used.Rows.Count; $r++) {
                for ($c = 1; $c -le $used.Columns.Count; $c++) {
                    $cell = $used.Cells.Item($r, $c)
                    $val  = $cell.Value2
                    if ($val -is [string] -and -not [string]::IsNullOrEmpty($val)) {
                        $mv = Invoke-TextMasking -Text $val
                        if ($mv -ne $val -and -not $DryRun) { $cell.Value2 = $mv }
                    }
                    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($cell) | Out-Null
                }
            }
            [System.Runtime.InteropServices.Marshal]::ReleaseComObject($used) | Out-Null
        }

        if (-not $DryRun) {
            Resolve-DestDir $DestPath
            $fmt = switch ([IO.Path]::GetExtension($DestPath).ToLower()) {
                '.xlsx' { 51 }; '.xlsm' { 52 }; '.xls' { 56 }; default { 51 }
            }
            $wb.SaveAs($DestPath, $fmt)
            Write-MaskLog "  -> 保存: $DestPath"
        }
    }
    finally {
        if ($wb)    { try { $wb.Close($false) }   catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($wb)    | Out-Null }
        if ($excel) { try { $excel.Quit() }        catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($excel) | Out-Null }
        [GC]::Collect(); [GC]::WaitForPendingFinalizers()
    }
}

# ==============================================================================
#  PowerPoint ファイル処理
# ==============================================================================
function Invoke-PowerPointFileMasking {
    param([string]$FilePath, [string]$DestPath)
    Write-MaskLog "PowerPoint: $(Split-Path $FilePath -Leaf)"
    $ppt = $null; $pres = $null
    try {
        $ppt  = New-Object -ComObject PowerPoint.Application
        $pres = $ppt.Presentations.Open($FilePath, $true, $false, $false)

        foreach ($slide in $pres.Slides) {
            foreach ($shape in $slide.Shapes) {
                if ($shape.HasTextFrame) {
                    $tf = $shape.TextFrame
                    $paraCount = $tf.TextRange.Paragraphs().Count
                    for ($pi = 1; $pi -le $paraCount; $pi++) {
                        $para = $tf.TextRange.Paragraphs($pi)
                        $t    = $para.Text
                        if (-not [string]::IsNullOrWhiteSpace($t)) {
                            $m = Invoke-TextMasking -Text $t
                            if ($m -ne $t -and -not $DryRun) { $para.Text = $m }
                        }
                    }
                }
            }

            # ノート
            if ($slide.HasNotesPage) {
                foreach ($shape in $slide.NotesPage.Shapes) {
                    if ($shape.HasTextFrame) {
                        $t = $shape.TextFrame.TextRange.Text
                        if (-not [string]::IsNullOrWhiteSpace($t)) {
                            $m = Invoke-TextMasking -Text $t
                            if ($m -ne $t -and -not $DryRun) { $shape.TextFrame.TextRange.Text = $m }
                        }
                    }
                }
            }
        }

        if (-not $DryRun) {
            Resolve-DestDir $DestPath
            $pres.SaveAs($DestPath)
            Write-MaskLog "  -> 保存: $DestPath"
        }
    }
    finally {
        if ($pres) { try { $pres.Close() } catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pres) | Out-Null }
        if ($ppt)  { try { $ppt.Quit()   } catch {}; [System.Runtime.InteropServices.Marshal]::ReleaseComObject($ppt)  | Out-Null }
        [GC]::Collect(); [GC]::WaitForPendingFinalizers()
    }
}

# ==============================================================================
#  出力パス解決
# ==============================================================================
function Resolve-OutputFilePath {
    param([string]$InputAbsPath)
    if ([string]::IsNullOrEmpty($OutputPath)) {
        return $InputAbsPath
    }
    if (Test-Path $OutputPath -PathType Container) {
        return Join-Path $OutputPath (Split-Path $InputAbsPath -Leaf)
    }
    return $OutputPath
}

# ==============================================================================
#  ファイル単体処理
# ==============================================================================
function Invoke-FileMasking {
    param([string]$FilePath)

    $absPath  = (Resolve-Path $FilePath).Path
    $destPath = Resolve-OutputFilePath $absPath

    # 上書きの場合はバックアップ作成
    if ($absPath -eq $destPath -and -not $DryRun) {
        $bak = "$absPath.$(Get-Date -Format 'yyyyMMddHHmmss').bak"
        Copy-Item $absPath $bak -Force
        Write-MaskLog "  バックアップ: $(Split-Path $bak -Leaf)"
    }

    $ext = [IO.Path]::GetExtension($FilePath).ToLower()
    switch ($ext) {
        { $_ -in '.doc','.docx','.docm' } {
            Invoke-WordFileMasking -FilePath $absPath -DestPath $destPath
        }
        { $_ -in '.xls','.xlsx','.xlsm','.xlsb' } {
            Invoke-ExcelFileMasking -FilePath $absPath -DestPath $destPath
        }
        { $_ -in '.ppt','.pptx','.pptm' } {
            Invoke-PowerPointFileMasking -FilePath $absPath -DestPath $destPath
        }
        { $_ -in '.txt','.csv','.tsv','.log','.xml','.json','.html','.htm',
                  '.md','.yaml','.yml','.ini','.conf','.config','.sql','.ps1','.sh' } {
            Invoke-TextFileMasking -FilePath $absPath -DestPath $destPath
        }
        default {
            Write-MaskLog "未対応の拡張子をスキップ: $ext ($FilePath)" 'WARN'
        }
    }
}

# ==============================================================================
#  エントリポイント
# ==============================================================================
if (-not (Test-Path $Path)) {
    Write-Error "パスが見つかりません: $Path"
    exit 1
}

$resolvedPath = (Resolve-Path $Path).Path

if ($DryRun) {
    Write-MaskLog '=== DRY RUN モード --- ファイルは一切変更されません ==='
}

# 処理対象ファイル収集
$officeExts = '*.doc','*.docx','*.docm','*.xls','*.xlsx','*.xlsm','*.xlsb','*.ppt','*.pptx','*.pptm'
$textExts   = '*.txt','*.csv','*.tsv','*.log','*.xml','*.json','*.html','*.htm',
              '*.md','*.yaml','*.yml','*.ini','*.conf','*.config','*.sql','*.ps1','*.sh'

$targets = if (Test-Path $resolvedPath -PathType Container) {
    $gciParams = @{
        Path    = $resolvedPath
        Include = ($officeExts + $textExts)
        File    = $true
    }
    if ($Recurse) { $gciParams['Recurse'] = $true }
    Get-ChildItem @gciParams
} else {
    @(Get-Item $resolvedPath)
}

Write-MaskLog "対象ファイル: $($targets.Count) 件"

$okCount = 0
$ngCount = 0

foreach ($f in $targets) {
    try {
        Invoke-FileMasking -FilePath $f.FullName
        $okCount++
    }
    catch {
        Write-MaskLog "エラー [$($f.Name)]: $_" 'ERROR'
        $ngCount++
    }
}

Write-MaskLog "=== 完了: 成功 $okCount 件 / エラー $ngCount 件 ==="
