# AI agent 修改 .pitchnet 的規則

這份規則給直接修改 .pitchnet 音符數字的 AI agent 使用。
目前只涵蓋「改音符音高」；其他修改都在第 6 節「待測」，驗證前不要做。
第 1～5 節每一條都有實測根據，根據寫在第 7 節。

## 1. 為什麼一定要執行 --apply

```
1. .pitchnet 裡存了整首播放用的聲音（waveform）。
   打開檔案時直接播放它，不會依照音符重新合成。
2. 只改音符數字，播放的聲音不會變。
3. WarpRender --apply 會重新合成有改到的音符，寫回 waveform 後存檔。
```

## 2. 工作流程

```
1. 開始修改前，先把要改的檔案複製一份，當作這一輪的「原檔」。
   例：cp song.pitchnet song.before.pitchnet
2. 修改另一份（「新檔」）的音符，依照第 3 節。原檔不要動。
3. 執行：
   WarpRender 新檔.pitchnet 輸出.pitchnet --apply --before 原檔.pitchnet
   1) WarpRender 位置：PitchNet/out/dev/tests/WarpRender
   2) 沒給 --before 時：印出用法、結束碼 2、不產生輸出檔。
   3) 成功時：結束碼 0，最後一行是「synthesized N notes, wrote 輸出檔路徑」。
4. 依照第 4 節檢查結果。
5. 把「輸出.pitchnet」交給使用者。
6. 連續修改多輪時，每一輪的原檔是上一輪的輸出檔。
   不要拿最初的檔案當原檔，否則前幾輪的音符會再合成一次，聲音會有細微差異。
```

## 3. 怎麼改音高

```
1. 只改 rest 為 false 的音符。
2. 改 midiNote，並把 lastNonMacroMidiNote 設成相同的值。
3. pitchOffset 保持 0。最終音高 = midiNote + pitchOffset，
   兩個欄位都加 3 會變成移 6 個半音。
4. 單位是半音（MIDI 音高，60 = C4，可以有小數）。
   移調 n 個半音：midiNote = 目前的 midiNote + n。
5. 改回原音高：midiNote 設成 originalMidiNote。
6. 除了 midiNote 和 lastNonMacroMidiNote，其他欄位都不要改。
7. 用 JSON 函式庫讀進整份檔案，改完整份寫回。
```

## 4. 找音符與檢查結果

```
1. 時間換算：音符的 startFrame／endFrame 單位是格，1 格 = 512 個取樣，取樣率 44100 Hz。
   秒數 = 格數 × 512 ÷ 44100。
2. --apply 的輸出應該符合：
   1) 「changed frames [起,迄)」落在改過的音符範圍，前後各多幾格是音高曲線的過渡。
   2) 「note @起始格」列出的就是改過的音符。
   3) 「synthesized N notes」的 N 等於改過的音符數。
   4) 最後一個音符之後的空白格子也可能被列為不同，不影響結果。
3. 再用比對指令確認：
   WarpRender 原檔.pitchnet 輸出.pitchnet --diff
   每個改過的音符列一行，移 n 個半音的音符，最大差距約 n × 100 cents。
4. 結果不符合時，不要交出輸出檔；回報指令、完整輸出和改過的音符。
```

## 5. 適用範圍

```
1. 獨立版 App：已驗證。
2. Cubase（Insert 與 ARA）：待測，見第 6 節。
```

## 6. 待測

以下項目還沒驗證，驗證前 agent 不要做。

```
1. Cubase 用 Insert 和 ARA 打開輸出檔，播放是否為新音高；
   ARA 片段的時間位置對應是否正確。
2. 修改 volumeDb（音量）、formantShift（共振峰）後，--apply 的聲音是否正確。
3. 修改 tiltLeft、tiltRight、tiltPivot、vibrato、pitchDrift、
   smoothLeftFrames、smoothRightFrames 後，--apply 的聲音是否正確。
4. agent 自己切開音符的做法
   （analysisSpans、originalDeltaPitch、f0Values 的切法）。
5. 合併、新增、刪除音符。
6. lastNonMacroMidiNote 沒跟著改時，之後在 App 用音高中心工具會不會回到舊音高。
7. 改了第 3 節第 6 條以外的欄位會發生什麼事。
```

## 7. 驗證紀錄

```
1. 「打開檔案直接播放 waveform、只改數字聽不到」：Q9 根因調查實測。
2. 第 2 節第 3 步的用法與結束碼：2026-11-19 實際執行，
   沒給 --before 時結束碼 2、沒有輸出檔；成功時結束碼 0。
3. 改音高（煙幕專案，第 130 個音符 +3 半音，2026-11-19）：
   1) 只改 midiNote、只改 pitchOffset 兩種改法，輸出逐個取樣相同。
   2) waveform 只在這個音符的範圍改變（取樣 2,969,089～3,052,521，格 5799～5961），
      和「格數 × 512」的換算相符。
   3) 離線量 waveform 音高：比原檔高 304／297／306 cents。
   4) 兩個欄位都 +3：--diff 量到 600 cents。
4. 獨立版 App：打開輸出檔，音符從 D4 顯示到 F4，其他音符不變。
5. 第二輪（以上一輪的輸出當原檔，同時改 midiNote 和 lastNonMacroMidiNote）：
   第 130 個音符改回 originalMidiNote、第 221 個音符 +2 半音。
   重算 2 個，--diff 300.03／200.00 cents；
   跟最初的原檔比，第 130 個音符的聲音回到逐個取樣相同。
6. 拿最初的檔案處理已處理過的檔案：音符再合成一次，波形最大差 -37.8 dBFS
   （Jingle bells 實驗）。
7. 最後一個音符之後的格子被列為不同、不影響結果：Jingle bells 實驗。
```
