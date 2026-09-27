import { X } from 'lucide-react';
import { Button } from '../../ui/button';
import { cn } from '../../ui/utils';
import type { ConvRecord } from '../types';

export type SidebarProps = {
  conversations: Array<ConvRecord> | null;
  loadError: string | null;
  onSelect: (id: string) => void;
  onDelete: (id: string) => void;
  onNew: () => void;
  onRetryLoad: () => void;
  /** Hide the drawer-only close control. */
  onClose?: () => void;
};

const ConversationList = ({ conversations, loadError, onSelect, onDelete, onRetryLoad }: SidebarProps) => {
  if (loadError !== null) {
    return (
      <div className="px-3 py-5 text-center font-mono text-xs text-faint">
        {loadError}
        <div className="mt-2.5">
          <Button type="button" variant="primaryOutline" size="sm" onClick={onRetryLoad} aria-label="Retry loading conversations">
            Retry
          </Button>
        </div>
      </div>
    );
  }
  if (conversations === null) {
    return <div className="px-3 py-5 text-center font-mono text-xs text-faint">Loading conversations…</div>;
  }
  if (conversations.length === 0) {
    return (
      <div className="px-3 py-5 text-center font-mono text-xs text-faint">
        No conversations yet.
        <br />
        Use <strong className="font-medium text-muted-foreground">+ New</strong> to start.
      </div>
    );
  }
  return (
    <div role="list" aria-label="Conversations">
      {conversations.map(function (conversation) {
        const label = conversation.title ?? 'New chat';
        return (
          <div
            key={conversation.id}
            role="listitem"
            className={cn(
              'mb-0.5 flex items-center gap-2 rounded-md border border-transparent px-3 py-2.5 transition-colors hover:bg-muted',
              conversation.active && 'border-primary/25 bg-primary/10',
            )}
          >
            <button
              type="button"
              onClick={function () { onSelect(conversation.id); }}
              aria-label={label}
              aria-current={conversation.active ? 'true' : undefined}
              className="flex min-w-0 flex-1 items-center rounded-sm text-start"
            >
              <span className={cn('flex-1 truncate text-sm', conversation.active ? 'text-foreground' : 'text-muted-foreground')} title={conversation.title}>
                {label}
              </span>
            </button>
            <button
              type="button"
              onClick={function () { onDelete(conversation.id); }}
              aria-label={`Delete conversation: ${label}`}
              className="agave-reveal inline-flex size-11 shrink-0 items-center justify-center rounded-xs text-muted-foreground transition-colors hover:bg-destructive/10 hover:text-destructive"
            >
              <X className="size-4" aria-hidden="true" />
            </button>
          </div>
        );
      })}
    </div>
  );
};

/** The conversation drawer body, shared by the desktop column and the mobile
 *  sheet so both surfaces stay identical. */
export const Sidebar = (props: SidebarProps) => (
  <div className="flex h-full w-full flex-col bg-card">
    <div className="flex shrink-0 items-center justify-between gap-2 border-b border-border px-3 py-3">
      <h2 className="font-mono text-xs font-medium text-faint">Chats</h2>
      <div className="flex shrink-0 items-center gap-1.5">
        {props.onClose ? (
          <Button type="button" size="iconSm" onClick={props.onClose} aria-label="Close sidebar" className="max-drawer:inline-flex">
            <X className="size-5" aria-hidden="true" />
          </Button>
        ) : null}
        <Button type="button" variant="primaryOutline" size="sm" onClick={props.onNew} aria-label="New conversation">
          + New
        </Button>
      </div>
    </div>
    <div className="agave-scroll flex-1 overflow-y-auto p-2">
      <ConversationList {...props} />
    </div>
  </div>
);