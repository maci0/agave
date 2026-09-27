import { useId } from 'react';
import { Button } from '../../ui/button';
import { Input } from '../../ui/input';
import { Label } from '../../ui/label';
import { Slider } from '../../ui/slider';
import { Textarea } from '../../ui/textarea';
import { fmtNum } from '../format';
import { MAX_TOKENS_MAX, MAX_TOKENS_MIN, clampMaxTokens, isMaxTokensValid } from '../storage';
import type { Sampling } from '../types';

const TEMPERATURE_MAX = 2;
const TOP_P_MAX = 1;

type SettingsPanelProps = {
  sampling: Sampling;
  onChange: (next: Sampling) => void;
  onClearSystem: () => void;
};

function SettingSlider({
  label,
  value,
  min,
  max,
  step,
  digits,
  hint,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  digits: number;
  hint: string;
  onChange: (next: number) => void;
}) {
  const id = useId();
  const shown = fmtNum(value, digits);
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <Label htmlFor={id}>{label}</Label>
        <span className="font-mono text-xs text-primary">{shown}</span>
      </div>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        onValueChange={function (next) { onChange(next[0] ?? value); }}
        thumbProps={{ id, 'aria-label': label, 'aria-valuetext': shown }}
      />
      <span className="mt-0.5 block font-mono text-2xs text-faint">{hint}</span>
    </div>
  );
}

export function SettingsPanel({ sampling, onChange, onClearSystem }: SettingsPanelProps) {
  const maxTokensId = useId();
  const systemId = useId();
  const maxTokensValid = isMaxTokensValid(sampling.maxTokens);

  return (
    <div
      id="settings-panel"
      role="region"
      aria-label="Sampling settings"
      className="mx-auto mb-3 max-w-prose rounded-lg border border-border bg-popover p-4"
    >
      <div className="mb-3 grid grid-cols-3 gap-4 max-drawer:grid-cols-1">
        <SettingSlider
          label="Temperature"
          value={sampling.temperature}
          min={0}
          max={TEMPERATURE_MAX}
          step={0.1}
          digits={1}
          hint="0 is focused, higher is more random"
          onChange={function (next) { onChange({ ...sampling, temperature: next }); }}
        />
        <SettingSlider
          label="Top-P"
          value={sampling.topP}
          min={0}
          max={TOP_P_MAX}
          step={0.05}
          digits={2}
          hint="1.0 considers all tokens"
          onChange={function (next) { onChange({ ...sampling, topP: next }); }}
        />
        <div>
          <Label htmlFor={maxTokensId} className="mb-1.5 block">
            Max Tokens
          </Label>
          <Input
            id={maxTokensId}
            type="number"
            inputMode="numeric"
            min={MAX_TOKENS_MIN}
            max={MAX_TOKENS_MAX}
            value={sampling.maxTokens}
            className="font-mono"
            aria-invalid={!maxTokensValid}
            aria-describedby={`${maxTokensId}-range${maxTokensValid ? '' : ` ${maxTokensId}-error`}`}
            onChange={function (event) { onChange({ ...sampling, maxTokens: event.target.value }); }}
            onKeyDown={function (event) {
              if (event.key === 'Enter') { event.preventDefault(); }
            }}
            onBlur={function () {
              if (maxTokensValid) {return;}
              onChange({ ...sampling, maxTokens: String(clampMaxTokens(sampling.maxTokens)) });
            }}
          />
          <span id={`${maxTokensId}-range`} className="mt-0.5 block font-mono text-2xs text-faint">
            {`${MAX_TOKENS_MIN}–${MAX_TOKENS_MAX}`}
          </span>
          {maxTokensValid ? null : (
            <span id={`${maxTokensId}-error`} role="alert" className="mt-1 block font-mono text-2xs text-destructive-foreground">
              {`Max tokens must be a whole number from ${MAX_TOKENS_MIN} to ${MAX_TOKENS_MAX}.`}
            </span>
          )}
        </div>
      </div>
      <div>
        <div className="mb-1.5 flex items-center justify-between">
          <Label htmlFor={systemId}>System Prompt</Label>
          <Button
            type="button"
            variant="ghost"
            size="sm"
            className="min-h-11 px-2 text-2xs hover:text-destructive-foreground"
            onClick={onClearSystem}
            aria-label="Clear system prompt"
          >
            Clear
          </Button>
        </div>
        <Textarea
          id={systemId}
          rows={2}
          spellCheck={false}
          autoCapitalize="off"
          dir="auto"
          placeholder="Optional system prompt..."
          value={sampling.system}
          className="min-h-10 max-h-30 resize-y"
          onChange={function (event) { onChange({ ...sampling, system: event.target.value }); }}
        />
      </div>
    </div>
  );
}
