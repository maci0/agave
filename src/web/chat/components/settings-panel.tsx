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

const SettingSlider = ({
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
}) => {
  const id = useId();
  const shown = fmtNum(value, digits);
  return (
    <div>
      <div className="mb-1.5 flex items-baseline justify-between gap-2">
        <Label htmlFor={id}>{label}</Label>
        <span className="font-mono text-xs text-primary">{shown}</span>
      </div>
      <Slider
        id={id}
        min={min}
        max={max}
        step={step}
        value={value}
        onValueChange={onChange}
        aria-label={label}
        aria-valuetext={shown}
      />
      <span className="mt-0.5 block font-mono text-2xs text-faint">{hint}</span>
    </div>
  );
};

/** The token budget: a numeric field with an inline range error. It is the one
 *  setting that is typed rather than dragged, so it validates as you type. */
const MaxTokensField = ({ sampling, onChange }: { sampling: Sampling; onChange: (next: Sampling) => void }) => {
  const fieldId = useId();
  const valid = isMaxTokensValid(sampling.maxTokens);
  return (
  <div>
    <Label htmlFor={fieldId} className="mb-1.5 block">
      Max tokens
    </Label>
    <Input
      id={fieldId}
      type="number"
      inputMode="numeric"
      min={MAX_TOKENS_MIN}
      max={MAX_TOKENS_MAX}
      value={sampling.maxTokens}
      font="mono"
      aria-invalid={!valid}
      aria-describedby={`${fieldId}-range${valid ? '' : ` ${fieldId}-error`}`}
      onChange={function (event) { onChange({ ...sampling, maxTokens: event.target.value }); }}
      onKeyDown={function (event) {
        if (event.key === 'Enter') { event.preventDefault(); }
      }}
      onBlur={function () {
        if (valid) {return;}
        onChange({ ...sampling, maxTokens: String(clampMaxTokens(sampling.maxTokens)) });
      }}
    />
    <span id={`${fieldId}-range`} className="mt-0.5 block font-mono text-2xs text-faint">
      {`${MAX_TOKENS_MIN}–${MAX_TOKENS_MAX}`}
    </span>
    {valid ? null : (
      <span id={`${fieldId}-error`} role="alert" className="mt-1 block font-mono text-2xs text-destructive-foreground">
        {`Max tokens must be a whole number from ${MAX_TOKENS_MIN} to ${MAX_TOKENS_MAX}.`}
      </span>
    )}
  </div>
  );
};

/** The system prompt, with the control that clears it from both stores. */
const SystemPromptField = ({ sampling, onChange, onClear }: {
  sampling: Sampling;
  onChange: (next: Sampling) => void;
  onClear: () => void;
}) => {
  const fieldId = useId();
  return (
    <div>
      <div className="mb-1.5 flex items-center justify-between">
        <Label htmlFor={fieldId}>System prompt</Label>
        <Button
          type="button"
          variant="plainDestructive"
          size="xs"
          onClick={onClear}
          aria-label="Clear system prompt"
        >
          Clear
        </Button>
      </div>
      <Textarea
        id={fieldId}
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
  );
};

export const SettingsPanel = ({ sampling, onChange, onClearSystem }: SettingsPanelProps) => (
  <div
    id="settings-panel"
    role="region"
    aria-label="Sampling settings"
    className="mx-auto mb-3 agave-measure rounded-lg border border-divider bg-popover p-4"
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
        label="Top-p"
        value={sampling.topP}
        min={0}
        max={TOP_P_MAX}
        step={0.05}
        digits={2}
        hint="1.0 considers all tokens"
        onChange={function (next) { onChange({ ...sampling, topP: next }); }}
      />
      <MaxTokensField sampling={sampling} onChange={onChange} />
    </div>
    <SystemPromptField sampling={sampling} onChange={onChange} onClear={onClearSystem} />
  </div>
);
